from __future__ import annotations

import argparse
import copy
import json
import os
from itertools import cycle
from pathlib import Path
from types import MethodType
from typing import Any

import cv2
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import functional as tvf


# This host sometimes inherits a dead local proxy. RF-DETR may need to download
# official pretrained weights, so prefer the working direct connection.
for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(key, None)

import rfdetr.datasets.coco as coco_mod  # noqa: E402
from rfdetr import RFDETRSmall  # noqa: E402
from rfdetr.utilities.box_ops import box_xyxy_to_cxcywh  # noqa: E402


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_CUSTOM_MEAN: tuple[float, ...] | None = None
_CUSTOM_STD: tuple[float, ...] | None = None
_USE_SPECTRAL_ADAPTER = False
_ADAPTER_ONLY = False


def _decode_multispectral(self: Any, image_id: int):
    path = Path(self.root) / self.coco.loadImgs(image_id)[0]["file_name"]
    file_bytes = np.fromfile(path, dtype=np.uint8)
    ok, frames = cv2.imdecodemulti(file_bytes, cv2.IMREAD_UNCHANGED)
    if not ok or not frames:
        raise RuntimeError(f"Failed to decode multispectral TIFF: {path}")
    image = np.stack(frames, axis=0)
    if image.ndim != 3:
        raise RuntimeError(f"Expected CxHxW TIFF stack, got {image.shape} for {path}")
    return tv_tensors.Image(torch.from_numpy(np.ascontiguousarray(image))), (1.0, 1.0)


def _convert_coco_detection(self: Any, image: Any, target: dict[str, Any]):
    if isinstance(image, torch.Tensor):
        h, w = map(int, image.shape[-2:])
    else:
        w, h = image.size

    image_id = torch.as_tensor([target["image_id"]])
    anno = [obj for obj in target["annotations"] if obj.get("iscrowd", 0) == 0]
    boxes = torch.as_tensor([obj["bbox"] for obj in anno], dtype=torch.float32).reshape(-1, 4)
    if len(boxes):
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(0, w)
        boxes[:, 1::2].clamp_(0, h)
    cat2label = self.cat2label
    labels = torch.as_tensor(
        [cat2label[obj["category_id"]] if cat2label is not None else obj["category_id"] for obj in anno],
        dtype=torch.int64,
    )
    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0]) if len(boxes) else torch.zeros(0, dtype=torch.bool)
    boxes = boxes[keep]
    labels = labels[keep]
    area = torch.as_tensor([obj.get("area", obj["bbox"][2] * obj["bbox"][3]) for obj in anno], dtype=torch.float32)
    iscrowd = torch.as_tensor([obj.get("iscrowd", 0) for obj in anno], dtype=torch.int64)
    out = {
        "boxes": boxes,
        "labels": labels,
        "image_id": image_id,
        "area": area[keep],
        "iscrowd": iscrowd[keep],
        "orig_size": torch.as_tensor([h, w]),
        "size": torch.as_tensor([h, w]),
    }
    return image, out


class DynamicNormalize:
    """Normalize multispectral tensors with cyclic ImageNet or fixed train-fold stats."""

    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> None:
        self.base_mean = tuple(mean)
        self.base_std = tuple(std)

    def __call__(self, image: torch.Tensor, target: dict[str, Any] | None = None):
        channels = int(image.shape[-3])
        if _CUSTOM_MEAN is not None and _CUSTOM_STD is not None:
            if len(_CUSTOM_MEAN) != channels or len(_CUSTOM_STD) != channels:
                raise ValueError(
                    f"Custom normalization has {len(_CUSTOM_MEAN)} means/{len(_CUSTOM_STD)} stds "
                    f"but image has {channels} channels"
                )
            mean = list(_CUSTOM_MEAN)
            std = list(_CUSTOM_STD)
        else:
            mean = [v for _, v in zip(range(channels), cycle(self.base_mean))]
            std = [v for _, v in zip(range(channels), cycle(self.base_std))]
        image = tvf.normalize(image, mean, std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            target["boxes"] = box_xyxy_to_cxcywh(target["boxes"]) / torch.tensor(
                [w, h, w, h], dtype=torch.float32
            )
        target["size"] = torch.as_tensor([h, w])
        return image, target


def configure_normalization(mode: str, stats_path: Path | None) -> None:
    global _CUSTOM_MEAN, _CUSTOM_STD
    if mode == "imagenet-cyclic":
        _CUSTOM_MEAN = None
        _CUSTOM_STD = None
        return
    if mode != "train-stats":
        raise ValueError(f"Unsupported normalization mode: {mode}")
    if stats_path is None:
        raise ValueError("--normalization train-stats requires --normalization-stats")
    data = json.loads(stats_path.read_text())
    mean = tuple(float(v) for v in data["mean"])
    std = tuple(float(v) for v in data["std"])
    if not mean or len(mean) != len(std) or any(v <= 0 for v in std):
        raise ValueError(f"Invalid normalization stats in {stats_path}")
    _CUSTOM_MEAN = mean
    _CUSTOM_STD = std


def install_identity_spectral_adapter(patch_embeddings: torch.nn.Module, channels: int) -> None:
    """Insert a registered identity 1x1 Conv before DINOv2 patch projection."""
    if hasattr(patch_embeddings, "spectral_adapter"):
        return
    adapter = torch.nn.Conv2d(channels, channels, kernel_size=1, bias=True)
    with torch.no_grad():
        adapter.weight.zero_()
        eye = torch.eye(channels, dtype=adapter.weight.dtype, device=adapter.weight.device)
        adapter.weight[:, :, 0, 0].copy_(eye)
        adapter.bias.zero_()
    patch_embeddings.add_module("spectral_adapter", adapter)

    def forward_with_adapter(self: Any, pixel_values: torch.Tensor) -> torch.Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        pixel_values = self.spectral_adapter(pixel_values)
        return self.projection(pixel_values).flatten(2).transpose(1, 2)

    patch_embeddings.forward = MethodType(forward_with_adapter, patch_embeddings)


def verify_identity_adapter(patch_embeddings: torch.nn.Module, channels: int) -> None:
    """Assert the identity adapter preserves patch-embedding output at initialization."""
    device = patch_embeddings.projection.weight.device
    dtype = patch_embeddings.projection.weight.dtype
    x = torch.rand(1, channels, 64, 64, device=device, dtype=dtype)
    with torch.no_grad():
        expected = patch_embeddings.projection(x).flatten(2).transpose(1, 2)
        actual = patch_embeddings(x)
    max_diff = float((expected - actual).abs().max().item())
    print(f"spectral_adapter_identity_maxdiff={max_diff:.9g}", flush=True)
    if max_diff > 1e-6:
        raise RuntimeError(f"Identity spectral adapter changed initial output: max_diff={max_diff}")


class SpectralAugment:
    """Weak train-only spectral perturbations applied on float tensors in [0, 1]."""

    def __init__(
        self,
        gain_min: float = 1.0,
        gain_max: float = 1.0,
        tilt: float = 0.0,
        noise_sigma: float = 0.0,
    ) -> None:
        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)
        self.tilt = float(tilt)
        self.noise_sigma = float(noise_sigma)

    def __call__(self, image: torch.Tensor, target: dict[str, Any] | None = None):
        if not torch.is_floating_point(image):
            raise TypeError("SpectralAugment expects a float tensor after ToDtype(scale=True)")

        gain = float(torch.empty(1).uniform_(self.gain_min, self.gain_max).item())
        slope = float(torch.empty(1).uniform_(-self.tilt, self.tilt).item()) if self.tilt > 0 else 0.0
        channels = int(image.shape[-3])
        if channels > 1:
            spectral_axis = torch.linspace(-1.0, 1.0, channels, dtype=image.dtype, device=image.device)
            spectral_axis = spectral_axis.view(channels, 1, 1)
            image = image * (gain * (1.0 + slope * spectral_axis))
        else:
            image = image * gain

        if self.noise_sigma > 0:
            image = image + torch.randn_like(image) * self.noise_sigma
        return image.clamp_(0.0, 1.0), target


def install_spectral_augmentation(
    gain_min: float,
    gain_max: float,
    tilt: float,
    noise_sigma: float,
) -> None:
    """Insert weak spectral augmentation immediately before normalization on train only."""

    def patch_builder(name: str) -> None:
        original = getattr(coco_mod, name)
        marker = f"_hotc_original_{name}"
        if hasattr(coco_mod, marker):
            original = getattr(coco_mod, marker)
        else:
            setattr(coco_mod, marker, original)

        def wrapped(image_set: str, *args: Any, **kwargs: Any):
            pipeline = original(image_set, *args, **kwargs)
            if image_set == "train":
                pipeline.transforms.insert(
                    -1,
                    SpectralAugment(
                        gain_min=gain_min,
                        gain_max=gain_max,
                        tilt=tilt,
                        noise_sigma=noise_sigma,
                    ),
                )
            return pipeline

        setattr(coco_mod, name, wrapped)

    patch_builder("make_coco_transforms")
    patch_builder("make_coco_transforms_square_div_64")


def install_multispectral_patches() -> None:
    coco_mod.CocoDetection._decode_image = _decode_multispectral
    coco_mod.ConvertCoco.__call__ = _convert_coco_detection
    coco_mod.Normalize = DynamicNormalize

    # RF-DETR's inference facade adapts the DINOv2 patch embedding to non-RGB
    # channels, but the Lightning training module rebuilds a fresh model and, in
    # v1.10.1, leaves that rebuilt backbone at 3 channels. Patch the training
    # module locally so it performs the same official 3->N adaptation *after*
    # task-specific pretrained weights are loaded. This also fixes evaluate(),
    # which constructs the same Lightning module internally.
    import rfdetr.training.module_model as module_model
    from rfdetr.inference import _adapt_input_conv

    cls = module_model.RFDETRModelModule
    if getattr(cls, "_hotc_multispectral_patched", False):
        return

    original_init = cls.__init__

    def patched_init(self: Any, model_config: Any, train_config: Any) -> None:
        target_channels = int(getattr(model_config, "num_channels", 3))
        deferred_state = None
        adapter_in_checkpoint = False
        init_config = model_config

        # RF-DETR v1.10.1 can adapt a 3-channel checkpoint to N channels, but it
        # cannot directly load an already-N-channel checkpoint because model
        # construction still starts from a 3-channel patch projection.  For a
        # same-channel local checkpoint, defer weight loading until after the
        # projection has been adapted below.
        pretrain_weights = getattr(model_config, "pretrain_weights", None)
        if target_channels != 3 and pretrain_weights:
            pretrain_path = Path(str(pretrain_weights))
            if pretrain_path.exists():
                checkpoint = torch.load(pretrain_path, map_location="cpu", weights_only=False)
                state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
                projection_key = "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight"
                if isinstance(state, dict) and projection_key in state:
                    checkpoint_channels = int(state[projection_key].shape[1])
                    if checkpoint_channels == target_channels:
                        deferred_state = state
                        adapter_in_checkpoint = any("spectral_adapter." in key for key in state)
                        if hasattr(model_config, "model_copy"):
                            init_config = model_config.model_copy(update={"pretrain_weights": None})
                        else:
                            init_config = copy.deepcopy(model_config)
                            init_config.pretrain_weights = None

        original_init(self, init_config, train_config)
        if target_channels == 3:
            return

        backbone = self.model.backbone[0]
        patch_embeddings = backbone.encoder.encoder.embeddings.patch_embeddings
        projection = patch_embeddings.projection
        if int(projection.in_channels) != target_channels:
            new_projection = copy.deepcopy(projection)
            new_projection.in_channels = target_channels
            new_weight = _adapt_input_conv(target_channels, projection.weight)
            new_projection.weight = torch.nn.Parameter(new_weight)
            new_projection.weight.requires_grad = projection.weight.requires_grad
            patch_embeddings.projection = new_projection
        patch_embeddings.num_channels = target_channels

        if _USE_SPECTRAL_ADAPTER:
            install_identity_spectral_adapter(patch_embeddings, target_channels)

        if deferred_state is not None:
            from rfdetr.models.weights import interpolate_position_embeddings

            interpolate_position_embeddings(
                deferred_state,
                int(getattr(model_config, "positional_encoding_size")),
            )
            incompatible = self.model.load_state_dict(deferred_state, strict=False)
            allowed_missing = {
                "backbone.0.encoder.encoder.embeddings.patch_embeddings.spectral_adapter.weight",
                "backbone.0.encoder.encoder.embeddings.patch_embeddings.spectral_adapter.bias",
            }
            missing = [key for key in incompatible.missing_keys if key not in allowed_missing]
            if missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "Deferred multispectral checkpoint load was incomplete: "
                    f"missing={missing[:10]} "
                    f"unexpected={incompatible.unexpected_keys[:10]}"
                )

        if _USE_SPECTRAL_ADAPTER and not adapter_in_checkpoint:
            verify_identity_adapter(patch_embeddings, target_channels)
            if _ADAPTER_ONLY:
                for parameter in self.model.parameters():
                    parameter.requires_grad = False
                for parameter in patch_embeddings.spectral_adapter.parameters():
                    parameter.requires_grad = True
                trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                print(f"adapter_only_trainable_params={trainable}", flush=True)

    cls.__init__ = patched_init
    cls._hotc_multispectral_patched = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("prepared/rfdetr_hsi16"))
    parser.add_argument("--output", type=Path, default=Path("runs/rfdetr_hsi16_small"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--channels", type=int, default=16, choices=(3, 16))
    parser.add_argument(
        "--pretrain",
        type=Path,
        default=None,
        help="Optional RF-DETR checkpoint to use instead of the official Small starter weights.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1.5e-4)
    parser.add_argument("--square", action="store_true", help="Force square resize; otherwise preserve aspect ratio")
    parser.add_argument(
        "--fixed-scale",
        action="store_true",
        help="Disable RF-DETR multi-scale training and keep the requested short-side resolution fixed.",
    )
    parser.add_argument("--spectral-gain", type=float, nargs=2, metavar=("MIN", "MAX"), default=None)
    parser.add_argument("--spectral-tilt", type=float, default=0.0)
    parser.add_argument("--spectral-noise", type=float, default=0.0)
    parser.add_argument(
        "--normalization",
        choices=("imagenet-cyclic", "train-stats"),
        default="imagenet-cyclic",
    )
    parser.add_argument("--normalization-stats", type=Path, default=None)
    parser.add_argument("--spectral-adapter", action="store_true")
    parser.add_argument(
        "--adapter-only",
        action="store_true",
        help="Freeze the detector and train only the identity-initialized 1x1 spectral adapter.",
    )
    parser.add_argument("--smoke-dataset-only", action="store_true")
    args = parser.parse_args()

    global _USE_SPECTRAL_ADAPTER, _ADAPTER_ONLY
    _USE_SPECTRAL_ADAPTER = bool(args.spectral_adapter or args.adapter_only)
    _ADAPTER_ONLY = bool(args.adapter_only)
    configure_normalization(args.normalization, args.normalization_stats)

    if args.channels != 3:
        install_multispectral_patches()

    if args.spectral_gain is not None or args.spectral_tilt > 0 or args.spectral_noise > 0:
        gain_min, gain_max = args.spectral_gain or (1.0, 1.0)
        if not (0 < gain_min <= gain_max):
            raise ValueError("--spectral-gain requires 0 < MIN <= MAX")
        install_spectral_augmentation(gain_min, gain_max, args.spectral_tilt, args.spectral_noise)

    # Constructing the model also exercises the official 3ch->16ch pretrained
    # patch-embedding adaptation path.
    model_kwargs: dict[str, Any] = {
        "num_channels": args.channels,
        "resolution": args.resolution,
    }
    deferred_outer_pretrain: str | None = None
    if args.pretrain is not None:
        pretrain_path = str(args.pretrain.resolve())
        checkpoint_channels = None
        if args.channels != 3 and args.pretrain.exists():
            checkpoint = torch.load(args.pretrain, map_location="cpu", weights_only=False)
            state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
            projection_key = "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight"
            if isinstance(state, dict) and projection_key in state:
                checkpoint_channels = int(state[projection_key].shape[1])
        if checkpoint_channels == args.channels and args.channels != 3:
            model_kwargs["pretrain_weights"] = None
            deferred_outer_pretrain = pretrain_path
        else:
            model_kwargs["pretrain_weights"] = pretrain_path
    model = RFDETRSmall(**model_kwargs)
    if deferred_outer_pretrain is not None:
        model.model_config.pretrain_weights = deferred_outer_pretrain
    print(f"model channels={model.model_config.num_channels} resolution={model.model_config.resolution}")
    print(f"means={len(model.means)} stds={len(model.stds)}")

    if args.smoke_dataset_only:
        from types import SimpleNamespace
        from rfdetr.datasets.coco import build_roboflow_from_coco

        cfg = SimpleNamespace(
            dataset_dir=str(args.dataset),
            square_resize_div_64=args.square,
            segmentation_head=False,
            multi_scale=False,
            expanded_scales=False,
            do_random_resize_via_padding=False,
            patch_size=model.model_config.patch_size,
            num_windows=model.model_config.num_windows,
            use_grouppose_keypoints=False,
            aug_config={},
            scale_jitter=False,
            augmentation_backend="cpu",
        )
        ds = build_roboflow_from_coco("train", cfg, args.resolution)
        image, target = ds[0]
        print(
            "dataset sample",
            tuple(image.shape),
            image.dtype,
            float(image.min()),
            float(image.max()),
            "boxes",
            tuple(target["boxes"].shape),
            "labels",
            target["labels"][:10].tolist(),
        )
        return

    args.output.mkdir(parents=True, exist_ok=True)
    print(
        f"optimization batch_size={args.batch} grad_accum_steps={args.grad_accum} "
        f"devices=1 effective_batch={args.batch * args.grad_accum}",
        flush=True,
    )
    model.train(
        dataset_dir=str(args.dataset),
        output_dir=str(args.output),
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        resolution=args.resolution,
        square_resize_div_64=args.square,
        num_workers=args.workers,
        aug_config={},
        scale_jitter=False,
        warmup_epochs=1.0,
        lr_scheduler="cosine",
        multi_scale=not args.fixed_scale,
        expanded_scales=not args.fixed_scale,
        checkpoint_interval=1,
        device=args.device,
    )


if __name__ == "__main__":
    main()
