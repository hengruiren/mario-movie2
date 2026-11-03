import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stage1"))

from prototype_patch.assets import (extract_class_features, extract_node_image_features,
                                   extract_node_text_features, extract_patch_features)
from prototype_patch.data import load_data


def main():
    parser = argparse.ArgumentParser(description="Extract frozen Llama class features and CLIP training patches")
    parser.add_argument("--mode", choices=("all", "classes", "patches", "node-text", "node-image"), default="all")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--movies-subset", default=str(ROOT / "splits/Movies/stage2_train20_seed42.json"))
    parser.add_argument("--class-descriptions", default=str(ROOT / "configs/movies_classes.json"))
    parser.add_argument("--llama-model", default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    parser.add_argument("--llama-revision", default="main")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch16")
    parser.add_argument("--clip-revision", default="main")
    parser.add_argument("--image-dir", default="")
    parser.add_argument("--image-extension", default=".jpg")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--llama-batch-size", type=int, default=1)
    parser.add_argument("--clip-batch-size", type=int, default=32)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.llama_batch_size < 1 or args.clip_batch_size < 1:
        parser.error("Extraction batch sizes must be positive")
    data = load_data(args.data_root, args.movies_subset)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    image_dir = args.image_dir or str(data.directory / "MoviesImages")
    if args.mode in ("all", "classes"):
        extract_class_features(data, args.class_descriptions, args.llama_model, output / "class_features.pt",
                                args.llama_revision, args.device, args.llama_batch_size,
                                args.dtype, args.local_files_only)
    if args.mode in ("all", "patches"):
        extract_patch_features(data, image_dir, args.clip_model, output / "clip_patches.pt",
                                args.clip_revision, args.device, args.clip_batch_size,
                                args.image_extension, args.local_files_only)
    if args.mode == "node-text":
        extract_node_text_features(data, args.llama_model, output / "Movies_llama_text_512_mean.npy",
                                    args.llama_revision, args.device, args.llama_batch_size,
                                    args.dtype, args.local_files_only)
    if args.mode == "node-image":
        extract_node_image_features(data, image_dir, args.llama_model, output / "Movies_llama_visual.npy",
                                     args.llama_revision, args.device, args.dtype,
                                     args.image_extension, args.local_files_only)


if __name__ == "__main__":
    main()
