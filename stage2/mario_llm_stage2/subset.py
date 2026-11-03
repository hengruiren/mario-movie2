import hashlib
import json
import math
from pathlib import Path

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stratified_targets(indices, labels, seed):
    indices = np.asarray(indices, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    classes, counts = np.unique(labels[indices], return_counts=True)
    total = int(math.floor(len(indices) * 0.2 + 0.5))
    if total == 0:
        raise ValueError("Training split is too small to select 20% of its targets")
    quotas = counts * total / len(indices)
    allocation = np.floor(quotas).astype(int)
    order = sorted(range(len(classes)), key=lambda i: (-(quotas[i] - allocation[i]), int(classes[i])))
    for i in order[:total - int(allocation.sum())]:
        allocation[i] += 1
    rng = np.random.RandomState(seed)
    selected = []
    for cls, count in zip(classes, allocation):
        candidates = np.sort(indices[labels[indices] == cls])
        selected.extend(rng.permutation(candidates)[:count].tolist())
    return sorted(selected)


def make_manifest(labels, dataset_dir, subset_seed=42):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    n = len(labels)
    ids = np.random.RandomState(42).permutation(n)
    train_end, val_end = int(n * 0.6), int(n * 0.6) + int(n * 0.2)
    splits = dict(zip(("train", "val", "test"),
                      (ids[:train_end], ids[train_end:val_end], ids[val_end:])))
    targets = stratified_targets(splits["train"], labels, subset_seed)
    classes, counts = np.unique(labels[targets], return_counts=True)
    return {
        "version": 2,
        "protocol": "stage2_train_only_20pct",
        "dataset": "Movies",
        "num_nodes": n,
        "split_seed": 42,
        "subset_seed": subset_seed,
        "fraction": 0.2,
        "label_sha256": hashlib.sha256(np.asarray(labels, dtype="<i8").tobytes()).hexdigest(),
        "source_sha256": {name: file_sha256(Path(dataset_dir) / name)
                          for name in ("Movies.csv", "MoviesGraph.pt")},
        "splits": {name: values.tolist() for name, values in splits.items()},
        "targets": {"train": targets},
        "class_counts": {"train": {str(int(c)): int(k) for c, k in zip(classes, counts)}},
    }


def save_manifest(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Existing Movies manifest differs; choose a new output file")
        return
    with path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_manifest(path, dataset, labels, dataset_dir):
    if dataset != "Movies":
        raise ValueError("--movies-subset is supported only for Movies")
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if (manifest.get("version") != 2
            or manifest.get("protocol") != "stage2_train_only_20pct"
            or type(manifest.get("subset_seed")) is not int):
        raise ValueError("Expected a Stage-2-only Movies 20% manifest")
    expected = make_manifest(labels, dataset_dir, manifest["subset_seed"])
    if manifest != expected:
        raise ValueError("Movies manifest does not match the dataset or fixed 20% protocol")
    return manifest
