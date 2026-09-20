from __future__ import annotations

import argparse
from pathlib import Path


def replace_once_or_verify(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new in text:
        print(f"already patched {path}")
        return
    if old in text:
        path.write_text(text.replace(old, new))
        print(f"patched {path}")
        return
    raise RuntimeError(f"Expected patch anchor not found in {path}: {old[:80]!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch official IDEA-Research/DINO for modern PyTorch and HODC training."
    )
    parser.add_argument("--dino-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.dino_root.resolve()

    cuda = root / "models/dino/ops/src/cuda/ms_deform_attn_cuda.cu"
    header = root / "models/dino/ops/src/ms_deform_attn.h"
    main_py = root / "main.py"
    engine = root / "engine.py"

    for path in (cuda, header):
        text = path.read_text()
        substitutions = {
            "value.type().is_cuda()": "value.is_cuda()",
            "spatial_shapes.type().is_cuda()": "spatial_shapes.is_cuda()",
            "level_start_index.type().is_cuda()": "level_start_index.is_cuda()",
            "sampling_loc.type().is_cuda()": "sampling_loc.is_cuda()",
            "attn_weight.type().is_cuda()": "attn_weight.is_cuda()",
            "grad_output.type().is_cuda()": "grad_output.is_cuda()",
            "AT_DISPATCH_FLOATING_TYPES(value.type(),": (
                "AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),"
            ),
        }
        changed = False
        for old, new in substitutions.items():
            if old in text:
                text = text.replace(old, new)
                changed = True
        if changed:
            path.write_text(text)
            print(f"patched {path}")
        else:
            print(f"no CUDA API changes needed in {path}")

    replace_once_or_verify(
        main_py,
        "checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']",
        "with torch.serialization.safe_globals([argparse.Namespace]):\n"
        "            checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']",
    )

    replace_once_or_verify(
        engine,
        "    _cnt = 0\n"
        "    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):",
        "    _cnt = 0\n"
        "    accum_iter = max(1, int(getattr(args, \"accum_iter\", 1)))\n"
        "    optimizer.zero_grad()\n"
        "    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):",
    )

    old_optim = '''        # amp backward function
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
            lr_scheduler.step()'''
    new_optim = '''        # amp backward function with optional gradient accumulation
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
            lr_scheduler.step()'''
    replace_once_or_verify(engine, old_optim, new_optim)


if __name__ == "__main__":
    main()
