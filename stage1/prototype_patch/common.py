import contextlib
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def save_tensor(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_vectors(value, name, dimensions):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(dimensions):
        raise ValueError("{} must have shape {}".format(name, dimensions))
    value = value.detach().cpu().float()
    if not torch.isfinite(value).all() or (value.norm(dim=-1) <= 1e-12).any():
        raise ValueError("{} contains non-finite or zero vectors".format(name))
    return value


def model_identity(model_path, revision, model=None):
    path = Path(model_path)
    identity = {"model": str(model_path), "requested_revision": revision}
    if path.is_dir():
        identity["local_files"] = {name: file_sha256(path / name)
                                   for name in ("config.json", "generation_config.json", "tokenizer_config.json",
                                                "tokenizer.json", "special_tokens_map.json", "vocab.json", "merges.txt",
                                                "preprocessor_config.json", "processor_config.json",
                                                "model.safetensors.index.json", "pytorch_model.bin.index.json")
                                   if (path / name).is_file()}
        weights = sorted(set(path.glob("*.safetensors")) | set(path.glob("pytorch_model*.bin")))
        identity["weight_sha256"] = {weight.name: file_sha256(weight) for weight in weights}
    if model is not None:
        identity["resolved_commit"] = getattr(model.config, "_commit_hash", None)
    return identity


@contextlib.contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
