import argparse
import csv
import json
from pathlib import Path

import numpy as np


def link(source, destination):
    source = Path(source).resolve(strict=True)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() and destination.resolve() == source:
        return
    if destination.exists() or destination.is_symlink():
        raise ValueError("Input view path already exists: {}".format(destination))
    destination.symlink_to(source, target_is_directory=source.is_dir())


def prepare(data_root, work_dir, text_feature, image_feature=None):
    source = Path(data_root).resolve() / "Movies"
    work_dir = Path(work_dir).resolve()
    image_path = (Path(image_feature).resolve() if image_feature else
                  source / "ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy")
    text_path = Path(text_feature).resolve()
    with (source / "Movies.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for i, row in enumerate(rows):
        if int(row["id"]) != i:
            raise ValueError("CSV row IDs must match original node IDs")
    for path in (image_path, text_path):
        features = np.load(path, mmap_mode="r")
        if features.shape != (len(rows), 4096):
            raise ValueError("Expected [num_nodes, 4096] features: {} has {}".format(path, features.shape))
    view = work_dir / "data/Movies"
    link(source / "Movies.csv", view / "Movies.csv")
    link(source / "MoviesGraph.pt", view / "MoviesGraph.pt")
    link(image_path, view / "ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy")
    link(text_path, view / "TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy")
    stage1_source = Path(__file__).resolve().parents[1] / "stage1"
    link(stage1_source / "config.json", work_dir / "stage1/config.json")
    metadata = {
        "data_root": str(source.parent),
        "image_feature": str(image_path.resolve()),
        "text_feature": str(text_path),
        "stage1_code": str(stage1_source),
        "num_nodes": len(rows),
    }
    metadata_path = work_dir / "inputs.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError("Existing input metadata differs")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Prepare file links for unmodified official Stage 1")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--text-feature", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.data_root, args.work_dir, args.text_feature), indent=2))


if __name__ == "__main__":
    main()
