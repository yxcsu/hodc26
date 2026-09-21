from __future__ import annotations

import argparse
from pathlib import Path


def replace_once_or_verify(path: Path, old: str, new: str, marker: str) -> None:
    text = path.read_text()
    if marker in text:
        print(f"already patched {path}: {marker}")
        return
    if old not in text:
        raise RuntimeError(f"Patch anchor not found in {path}: {old[:120]!r}")
    path.write_text(text.replace(old, new, 1))
    print(f"patched {path}: {marker}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch official DINO with 19-channel HSI residual-stem support."
    )
    parser.add_argument("--dino-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.dino_root.resolve()

    transforms = root / "datasets/transforms.py"
    coco = root / "datasets/coco.py"
    backbone = root / "models/dino/backbone.py"
    main_py = root / "main.py"

    replace_once_or_verify(
        transforms,
        "from util.misc import interpolate\n",
        "from util.misc import interpolate\n\n\n"
        "def _hotc_image_size(image):\n"
        "    if isinstance(image, torch.Tensor):\n"
        "        return int(image.shape[-1]), int(image.shape[-2])\n"
        "    return image.size\n",
        "_hotc_image_size",
    )

    text = transforms.read_text()
    substitutions = {
        "    w, h = image.size\n": "    w, h = _hotc_image_size(image)\n",
        "    size = get_size(image.size, size, max_size)\n"
        "    rescaled_image = F.resize(image, size)\n":
            "    original_size = _hotc_image_size(image)\n"
            "    size = get_size(original_size, size, max_size)\n"
            "    rescaled_image = F.resize(image, size)\n",
        "    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))\n":
            "    rescaled_size = _hotc_image_size(rescaled_image)\n"
            "    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_size, original_size))\n",
        "    target[\"size\"] = torch.tensor(padded_image.size[::-1])\n":
            "    pw, ph = _hotc_image_size(padded_image)\n"
            "    target[\"size\"] = torch.tensor([ph, pw])\n",
        "        w = random.randint(self.min_size, min(img.width, self.max_size))\n"
        "        h = random.randint(self.min_size, min(img.height, self.max_size))\n":
            "        image_width, image_height = _hotc_image_size(img)\n"
            "        w = random.randint(self.min_size, min(image_width, self.max_size))\n"
            "        h = random.randint(self.min_size, min(image_height, self.max_size))\n",
        "        image_width, image_height = img.size\n":
            "        image_width, image_height = _hotc_image_size(img)\n",
        "class ToTensor(object):\n"
        "    def __call__(self, img, target):\n"
        "        return F.to_tensor(img), target\n":
            "class ToTensor(object):\n"
            "    def __call__(self, img, target):\n"
            "        if isinstance(img, torch.Tensor):\n"
            "            if img.dtype == torch.uint8:\n"
            "                img = img.float().div(255.0)\n"
            "            elif not torch.is_floating_point(img):\n"
            "                img = img.float()\n"
            "            return img, target\n"
            "        return F.to_tensor(img), target\n",
    }
    changed = False
    for old, new in substitutions.items():
        if old in text:
            text = text.replace(old, new)
            changed = True
    if changed:
        transforms.write_text(text)
        print(f"patched tensor transforms in {transforms}")

    replace_once_or_verify(
        coco,
        "import torch\nimport torch.utils.data\nimport torchvision\n",
        "import torch\nimport torch.utils.data\nimport torchvision\n"
        "import cv2\nimport numpy as np\n",
        "import cv2",
    )

    text = coco.read_text()
    old = '''class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, aux_target_hacks=None):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.aux_target_hacks = aux_target_hacks
'''
    new = '''class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, aux_target_hacks=None, multispectral_channels=3):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.aux_target_hacks = aux_target_hacks
        self.multispectral_channels = int(multispectral_channels)

    def _hotc_decode_multispectral(self, image_id):
        info = self.coco.loadImgs(image_id)[0]
        path = Path(self.root) / info["file_name"]
        file_bytes = np.fromfile(path, dtype=np.uint8)
        ok, frames = cv2.imdecodemulti(file_bytes, cv2.IMREAD_UNCHANGED)
        if not ok or not frames:
            raise RuntimeError(f"Failed to decode multispectral TIFF: {path}")
        image = np.stack(frames, axis=0)
        if image.ndim != 3 or image.shape[0] != self.multispectral_channels:
            raise RuntimeError(
                f"Expected {self.multispectral_channels}xHxW TIFF, got {image.shape}: {path}"
            )
        tensor = torch.from_numpy(np.ascontiguousarray(image))
        if tensor.dtype == torch.uint8:
            tensor = tensor.float().div_(255.0)
        elif not torch.is_floating_point(tensor):
            tensor = tensor.float()
            max_value = float(tensor.max().item())
            if max_value > 1.0:
                tensor.div_(max_value)
        return tensor
'''
    if "_hotc_decode_multispectral" not in text:
        if old not in text:
            raise RuntimeError("CocoDetection init anchor not found")
        text = text.replace(old, new, 1)

    old = '''        try:
            img, target = super(CocoDetection, self).__getitem__(idx)
        except:
            print("Error idx: {}".format(idx))
            idx += 1
            img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
'''
    new = '''        image_id = self.ids[idx]
        if self.multispectral_channels == 3:
            try:
                img, target = super(CocoDetection, self).__getitem__(idx)
            except Exception:
                print("Error idx: {}".format(idx))
                idx += 1
                image_id = self.ids[idx]
                img, target = super(CocoDetection, self).__getitem__(idx)
        else:
            img = self._hotc_decode_multispectral(image_id)
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            target = self.coco.loadAnns(ann_ids)
        target = {'image_id': image_id, 'annotations': target}
'''
    if "if self.multispectral_channels == 3:" not in text:
        if old not in text:
            raise RuntimeError("CocoDetection getitem anchor not found")
        text = text.replace(old, new, 1)

    old = "        w, h = image.size\n\n        image_id = target[\"image_id\"]\n"
    new = (
        "        if isinstance(image, torch.Tensor):\n"
        "            h, w = map(int, image.shape[-2:])\n"
        "        else:\n"
        "            w, h = image.size\n\n"
        "        image_id = target[\"image_id\"]\n"
    )
    tensor_size_marker = (
        "        if isinstance(image, torch.Tensor):\n"
        "            h, w = map(int, image.shape[-2:])\n"
        "        else:\n"
        "            w, h = image.size\n"
    )
    if tensor_size_marker not in text and old in text:
        text = text.replace(old, new, 1)

    old = '''    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
'''
    new = '''    multispectral_channels = int(getattr(args, "multispectral_channels", 3))
    if multispectral_channels == 19:
        base_mean = [0.485, 0.456, 0.406]
        base_std = [0.229, 0.224, 0.225]
        mean = [base_mean[i % 3] for i in range(16)] + base_mean
        std = [base_std[i % 3] for i in range(16)] + base_std
    elif multispectral_channels == 3:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise ValueError(f"Unsupported multispectral_channels={multispectral_channels}")

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std)
    ])
'''
    if 'multispectral_channels = int(getattr(args, "multispectral_channels", 3))' not in text:
        if old not in text:
            raise RuntimeError("normalize anchor not found")
        text = text.replace(old, new, 1)

    old = '''            return_masks=args.masks,
            aux_target_hacks=aux_target_hacks_list,
        )
'''
    new = '''            return_masks=args.masks,
            aux_target_hacks=aux_target_hacks_list,
            multispectral_channels=getattr(args, "multispectral_channels", 3),
        )
'''
    if "multispectral_channels=getattr(args" not in text:
        if old not in text:
            raise RuntimeError("dataset constructor anchor not found")
        text = text.replace(old, new, 1)
    coco.write_text(text)
    print(f"patched multispectral dataset in {coco}")

    replace_once_or_verify(
        backbone,
        "class Backbone(BackboneBase):\n",
        '''class HSIResidualConv1(nn.Module):
    """Behavior-preserving 19ch stem: RGB conv + zero-init HSI residual conv."""

    def __init__(self, rgb_conv: nn.Conv2d, hsi_channels: int = 16):
        super().__init__()
        self.rgb_conv = rgb_conv
        self.hsi_channels = int(hsi_channels)
        self.spectral_conv = nn.Conv2d(
            self.hsi_channels,
            rgb_conv.out_channels,
            kernel_size=rgb_conv.kernel_size,
            stride=rgb_conv.stride,
            padding=rgb_conv.padding,
            dilation=rgb_conv.dilation,
            groups=rgb_conv.groups,
            bias=False,
        )
        nn.init.zeros_(self.spectral_conv.weight)

    def forward(self, x):
        expected = self.hsi_channels + 3
        if x.shape[1] != expected:
            raise ValueError(f"HSI residual stem expected {expected} channels, got {x.shape[1]}")
        return self.rgb_conv(x[:, self.hsi_channels:]) + self.spectral_conv(x[:, :self.hsi_channels])


class Backbone(BackboneBase):
''',
        "class HSIResidualConv1",
    )

    text = backbone.read_text()
    old = '''        backbone = Backbone(args.backbone, train_backbone, args.dilation,   
                                return_interm_indices,   
                                batch_norm=FrozenBatchNorm2d)
        bb_num_channels = backbone.num_channels
'''
    new = '''        backbone = Backbone(args.backbone, train_backbone, args.dilation,
                                return_interm_indices,
                                batch_norm=FrozenBatchNorm2d)
        if getattr(args, "hsi_residual_stem", False):
            backbone.body.conv1 = HSIResidualConv1(backbone.body.conv1, hsi_channels=16)
            if getattr(args, "hsi_residual_only", False):
                for parameter in backbone.parameters():
                    parameter.requires_grad_(False)
                for parameter in backbone.body.conv1.spectral_conv.parameters():
                    parameter.requires_grad_(True)
        bb_num_channels = backbone.num_channels
'''
    if 'getattr(args, "hsi_residual_stem", False)' not in text:
        if old not in text:
            raise RuntimeError("ResNet build anchor not found")
        text = text.replace(old, new, 1)
        backbone.write_text(text)
        print(f"patched residual stem in {backbone}")

    text = main_py.read_text()
    old = '''    # build model
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)
'''
    new = '''    # build model
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)
    if getattr(args, "hsi_residual_only", False):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        spectral_parameters = []
        for name, parameter in model.named_parameters():
            if name.endswith("spectral_conv.weight"):
                parameter.requires_grad_(True)
                spectral_parameters.append(name)
        if not spectral_parameters:
            raise RuntimeError("hsi_residual_only requested but no spectral_conv parameters found")
        print("HSI Phase A trainable parameters:", spectral_parameters)
'''
    if 'HSI Phase A trainable parameters:' not in text:
        if old not in text:
            raise RuntimeError("model build anchor not found")
        text = text.replace(old, new, 1)
        main_py.write_text(text)
        print(f"patched Phase A freeze in {main_py}")

    old = '''        _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
'''
    new = '''        _clean_checkpoint = utils.clean_state_dict(checkpoint)
        if getattr(args, "hsi_residual_stem", False):
            conv1_key = "backbone.0.body.conv1.weight"
            residual_rgb_key = "backbone.0.body.conv1.rgb_conv.weight"
            if conv1_key in _clean_checkpoint and residual_rgb_key not in _clean_checkpoint:
                _clean_checkpoint = OrderedDict(_clean_checkpoint)
                _clean_checkpoint[residual_rgb_key] = _clean_checkpoint.pop(conv1_key)
        _tmp_st = OrderedDict({k:v for k, v in _clean_checkpoint.items() if check_keep(k, _ignorekeywordlist)})

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
'''
    if "_clean_checkpoint = utils.clean_state_dict(checkpoint)" not in text:
        if old not in text:
            raise RuntimeError("pretrain load anchor not found")
        text = text.replace(old, new, 1)
        main_py.write_text(text)
        print(f"patched checkpoint remap in {main_py}")

    for path in (transforms, coco, backbone, main_py):
        compile(path.read_text(), str(path), "exec")
    print("DINO HSI19 patch complete")


if __name__ == "__main__":
    main()
