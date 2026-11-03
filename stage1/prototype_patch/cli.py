import argparse
import json
import math
from pathlib import Path

from .common import ROOT


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Mario Stage 1: frozen Llama prototypes and CLIP patch distillation")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--text-feature", required=True)
    parser.add_argument("--image-feature", required=True)
    parser.add_argument("--class-cache", required=True)
    parser.add_argument("--patch-cache", required=True)
    parser.add_argument("--class-descriptions", default=str(ROOT / "configs/movies_classes.json"))
    parser.add_argument("--movies-subset", default=str(ROOT / "splits/Movies/stage2_train20_seed42.json"))
    parser.add_argument("--config", default=str(ROOT / "configs/prototype_patch.json"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-checkpoint", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--method", choices=("full", "class_only", "fine_only", "infonce_only"), default="full")
    for name in ("seed", "base-epochs", "base-batch-size", "finetune-epochs", "finetune-batch-size",
                 "class-only-epochs", "fine-ramp-epochs", "top-k"):
        parser.add_argument("--" + name, type=int, default=None)
    for name in ("lambda-class", "lambda-fine", "class-temperature", "finetune-lr"):
        parser.add_argument("--" + name, type=float, default=None)
    args = parser.parse_args(argv)
    settings = json.loads(Path(args.config).read_text())
    for key in settings:
        value = getattr(args, key, None)
        if value is not None:
            settings[key] = value
    if args.method in ("fine_only", "infonce_only"):
        settings["lambda_class"] = 0.0
    if args.method in ("class_only", "infonce_only"):
        settings["lambda_fine"] = 0.0
    if any(isinstance(value, (int, float)) and not math.isfinite(value) for value in settings.values()):
        parser.error("Numeric settings must be finite")
    positive = ("base_epochs", "base_batch_size", "base_lr", "finetune_epochs", "finetune_batch_size",
                "finetune_lr", "class_temperature", "top_k", "eval_batch_size", "probe_ridge")
    if any(settings[key] <= 0 for key in positive):
        parser.error("Epoch counts, batch sizes, learning rates, temperatures, top_k and probe ridge must be positive")
    if any(settings[key] < 0 for key in ("class_only_epochs", "fine_ramp_epochs", "lambda_class", "lambda_fine")):
        parser.error("Warmup, ramp and loss weights must be nonnegative")
    if settings["selection"] != "last":
        parser.error("This experiment exports the last checkpoint; selection must be 'last'")
    return args, settings


def main():
    args, settings = parse_args()
    print(json.dumps({"method": args.method, "resolved_settings": settings}, indent=2), flush=True)
    from .training import run
    run(args, settings)
