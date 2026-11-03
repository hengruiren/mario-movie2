"""Frozen full-image and 2x2 crop CLIP features, including unlabeled nodes."""

import gc

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .assets import cached_payload, image_inventory, patch_views
from .common import file_sha256, json_digest, model_identity, save_tensor, validate_vectors


@torch.no_grad()
def extract_all_clip(data, image_dir, model_path, output, revision="main", device="cuda",
                     batch_size=32, extension=".jpg", local_files_only=False):
    from PIL import Image, ImageOps

    if batch_size < 1:
        raise ValueError("CLIP batch size must be positive")
    ids = torch.arange(len(data.records))
    paths, inventory = image_inventory(image_dir, ids, extension)
    expected = dict(kind="clip_all_nodes_global_and_2x2", grid=[2, 2],
                    csv_sha256=file_sha256(data.directory / "Movies.csv"),
                    graph_sha256=file_sha256(data.directory / "MoviesGraph.pt"),
                    node_ids_sha256=json_digest(ids.tolist()), image_inventory_sha256=inventory,
                    model_identity=model_identity(model_path, revision),
                    preprocessing="EXIF-corrected RGB; CLIP default preprocessing per full image/crop",
                    crop_order="top_left,top_right,bottom_left,bottom_right")
    payload = cached_payload(output, expected)
    if payload is not None:
        validate_clip_cache(payload, data)
        return payload
    from transformers import CLIPModel, CLIPProcessor

    options = dict(revision=revision, local_files_only=local_files_only)
    processor = CLIPProcessor.from_pretrained(model_path, **options)
    model = CLIPModel.from_pretrained(model_path, **options).to(device).eval().requires_grad_(False)
    if model.config.projection_dim != 512 or model.config.vision_config.patch_size != 16:
        raise ValueError("Expected CLIP ViT-B/16 with projected dimension 512")
    global_features, patches = torch.empty(len(ids), 512), torch.empty(len(ids), 4, 512)
    per_batch = max(batch_size // 5, 1)
    for start in tqdm(range(0, len(ids), per_batch), desc="CLIP all-node images + patches"):
        selected_paths = paths[start:start + per_batch]
        views = []
        for path in selected_paths:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                views.extend([image.copy()] + patch_views(image))
        vectors = []
        for offset in range(0, len(views), batch_size):
            inputs = processor(images=views[offset:offset + batch_size], return_tensors="pt",
                               input_data_format="channels_last").to(device)
            vectors.append(F.normalize(model.get_image_features(**inputs).float(), dim=-1).cpu())
        features = torch.cat(vectors).reshape(len(selected_paths), 5, 512)
        global_features[start:start + len(selected_paths)] = features[:, 0]
        patches[start:start + len(selected_paths)] = features[:, 1:]
    payload = dict(global_features=global_features, patches=patches, node_ids=ids,
                   metadata=dict(expected, resolved_commit=getattr(model.config, "_commit_hash", None)))
    validate_clip_cache(payload, data)
    save_tensor(output, payload)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def validate_clip_cache(payload, data):
    metadata = payload.get("metadata", {})
    ids = torch.arange(len(data.records))
    expected = dict(kind="clip_all_nodes_global_and_2x2", grid=[2, 2],
                    csv_sha256=file_sha256(data.directory / "Movies.csv"),
                    graph_sha256=file_sha256(data.directory / "MoviesGraph.pt"),
                    node_ids_sha256=json_digest(ids.tolist()))
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("All-node CLIP cache provenance differs; re-extract for this dataset")
    if not torch.equal(payload.get("node_ids", torch.empty(0)), ids):
        raise ValueError("CLIP cache must cover ALL nodes in original node-ID order")
    validate_vectors(payload["global_features"], "CLIP full images", (len(ids), 512))
    validate_vectors(payload["patches"], "CLIP patches", (len(ids), 4, 512))
