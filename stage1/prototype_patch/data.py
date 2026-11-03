import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import torch

from .common import ROOT, file_sha256

sys.path.insert(0, str(ROOT))
from stage2.mario_llm_stage2.subset import load_manifest


@dataclass
class MoviesData:
    directory: Path
    records: list
    labels: torch.Tensor
    names: list
    manifest: dict
    train_ids: torch.Tensor
    val_ids: torch.Tensor
    test_ids: torch.Tensor
    text: torch.Tensor = None
    image: torch.Tensor = None


def load_data(data_root, manifest_path, text_path=None, image_path=None):
    directory = Path(data_root).resolve() / "Movies"
    with (directory / "Movies.csv").open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    if not records or any(int(row["id"]) != i for i, row in enumerate(records)):
        raise ValueError("Movies CSV rows must be in original node-ID order")
    labels = torch.tensor([int(row["label"]) for row in records], dtype=torch.long)
    manifest = load_manifest(manifest_path, "Movies", labels.numpy(), directory)
    names = []
    for category in range(int(labels.max()) + 1):
        values = {row["second_category"] for row in records if int(row["label"]) == category}
        if len(values) != 1:
            raise ValueError("Each class ID must have exactly one category name")
        names.append(values.pop())
    if len(names) != 20:
        raise ValueError("This implementation requires the 20 Movies classes")
    data = MoviesData(directory, records, labels, names, manifest,
                       torch.tensor(sorted(manifest["splits"]["train"])),
                       torch.tensor(manifest["splits"]["val"]),
                       torch.tensor(manifest["splits"]["test"]))
    for name, path in (("text", text_path), ("image", image_path)):
        if path is None:
            continue
        values = np.load(path, mmap_mode="r")
        if values.shape != (len(records), 4096):
            raise ValueError("{} features must have shape [N, 4096]".format(name))
        if not np.isfinite(values).all():
            raise ValueError("{} features contain non-finite values".format(name))
        setattr(data, name, torch.from_numpy(np.array(values, dtype=np.float32)))
    return data


def load_descriptions(path, names):
    records = json.loads(Path(path).read_text())
    if [row["id"] for row in records] != list(range(len(names))):
        raise ValueError("Class descriptions must be ordered by class ID")
    if [row["name"] for row in records] != names:
        raise ValueError("Class descriptions do not match Movies category IDs and names")
    if any(not row["description"].strip() for row in records):
        raise ValueError("Every class needs a semantic description")
    return records


def data_signature(data, manifest_path, text_path, image_path):
    return {
        "csv_sha256": file_sha256(data.directory / "Movies.csv"),
        "graph_sha256": file_sha256(data.directory / "MoviesGraph.pt"),
        "stage2_manifest_sha256": file_sha256(manifest_path),
        "text_feature_sha256": file_sha256(text_path),
        "image_feature_sha256": file_sha256(image_path),
        "stage1_training_scope": "full_training_split",
        "stage1_train_nodes": len(data.train_ids),
        "stage2_train_targets": len(data.manifest["targets"]["train"]),
        "validation_nodes": len(data.val_ids),
        "test_nodes": len(data.test_ids),
    }
