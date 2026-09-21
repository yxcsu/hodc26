#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/yangxiao/hotc-venv/bin/python}"
DINO_ROOT="${DINO_ROOT:-/home/yangxiao/.cache/hodc26/DINO}"
DINO_REPO="${DINO_REPO:-https://github.com/IDEA-Research/DINO.git}"
DINO_COMMIT="${DINO_COMMIT:-d84a491d41898b3befd8294d1cf2614661fc0953}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

mkdir -p "$(dirname "$DINO_ROOT")"
if [[ ! -d "$DINO_ROOT/.git" ]]; then
  git clone "$DINO_REPO" "$DINO_ROOT"
fi
git -C "$DINO_ROOT" fetch --all --tags
git -C "$DINO_ROOT" checkout "$DINO_COMMIT"

"$PYTHON" -m pip install \
  cython submitit termcolor addict "yapf==0.32.0" timm pycocotools

"$PYTHON" - "$DINO_ROOT" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])

# PyTorch 2.x compatibility for the old custom CUDA extension.
for path in [
    root / "models/dino/ops/src/cuda/ms_deform_attn_cuda.cu",
    root / "models/dino/ops/src/ms_deform_attn.h",
]:
    text = path.read_text()
    replacements = {
        "value.type().is_cuda()": "value.is_cuda()",
        "spatial_shapes.type().is_cuda()": "spatial_shapes.is_cuda()",
        "level_start_index.type().is_cuda()": "level_start_index.is_cuda()",
        "sampling_loc.type().is_cuda()": "sampling_loc.is_cuda()",
        "attn_weight.type().is_cuda()": "attn_weight.is_cuda()",
        "grad_output.type().is_cuda()": "grad_output.is_cuda()",
        "AT_DISPATCH_FLOATING_TYPES(value.type(),":
            "AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    path.write_text(text)

# PyTorch 2.6+ defaults torch.load(weights_only=True).  The official DINO
# checkpoint contains argparse.Namespace metadata, so allowlist exactly that
# type instead of disabling safe loading globally.
main_path = root / "main.py"
text = main_path.read_text()
old = "checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']"
new = (
    "with torch.serialization.safe_globals([argparse.Namespace]):\n"
    "            checkpoint = torch.load(args.pretrain_model_path, "
    "map_location='cpu')['model']"
)
if old in text:
    text = text.replace(old, new)
main_path.write_text(text)

# Gradient accumulation keeps the official effective batch size of 16 on a
# shared 24 GB GPU.  This patch is idempotent.
engine_path = root / "engine.py"
text = engine_path.read_text()
old = """    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):

        samples = samples.to(device)"""
new = """    _cnt = 0
    accum_iter = max(1, int(getattr(args, "accum_iter", 1)))
    optimizer.zero_grad()
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):

        samples = samples.to(device)"""
if old in text:
    text = text.replace(old, new)

old = """        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()"""
new = """        # amp backward function with optional gradient accumulation
        loss_for_backward = losses / accum_iter
        should_step = ((_cnt + 1) % accum_iter == 0) or ((_cnt + 1) == len(data_loader))
        if args.amp:
            scaler.scale(loss_for_backward).backward()
            if should_step:
                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss_for_backward.backward()
            if should_step:
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                optimizer.zero_grad()

        if args.onecyclelr and should_step:
            lr_scheduler.step()"""
if old in text:
    text = text.replace(old, new)
engine_path.write_text(text)
PY

"$PYTHON" -m py_compile "$DINO_ROOT/main.py" "$DINO_ROOT/engine.py"

pushd "$DINO_ROOT/models/dino/ops" >/dev/null
rm -rf build
CUDA_HOME="$CUDA_HOME" TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  "$PYTHON" setup.py build install
popd >/dev/null

echo "Official DINO ready at: $DINO_ROOT"
echo "Commit: $(git -C "$DINO_ROOT" rev-parse HEAD)"
