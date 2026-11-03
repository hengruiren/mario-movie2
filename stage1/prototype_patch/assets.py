import gc
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .common import file_sha256, json_digest, model_identity, save_tensor, validate_vectors, write_json
from .data import load_descriptions


def mean_pool(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def patch_views(image):
    from PIL import ImageOps
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width < 2 or height < 2:
        raise ValueError("Image is too small for a 2x2 crop grid")
    return [image.crop((width * col // 2, height * row // 2,
                        width * (col + 1) // 2, height * (row + 1) // 2))
            for row in range(2) for col in range(2)]


def cached_payload(path, expected):
    path = Path(path)
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Existing cache settings differ: {}".format(path))
    return payload


@torch.no_grad()
def extract_class_features(data, descriptions_path, model_path, output, revision="main",
                           device="cuda", batch_size=1, dtype="bfloat16", local_files_only=False):
    records = load_descriptions(descriptions_path, data.names)
    expected = {
        "kind": "llama_class_text",
        "csv_sha256": file_sha256(data.directory / "Movies.csv"),
        "descriptions_sha256": file_sha256(descriptions_path),
        "model_identity": model_identity(model_path, revision),
        "pooling": "last_hidden_attention_masked_mean",
        "max_length": 256,
        "chat_template": False,
        "dtype": dtype,
    }
    payload = cached_payload(output, expected)
    if payload is not None:
        validate_vectors(payload["features"], "Class features", (20, 4096))
        return payload
    texts = ["{}. {}".format(row["name"], row["description"]) for row in records]
    features, resolved = llama_text_features(texts, model_path, revision, device,
                                            batch_size, dtype, local_files_only)
    validate_vectors(features, "Class features", (20, 4096))
    payload = {"features": features, "class_ids": torch.arange(20),
               "class_names": data.names, "texts": texts,
               "metadata": dict(expected, resolved_commit=resolved)}
    save_tensor(output, payload)
    return payload


@torch.no_grad()
def llama_text_features(texts, model_path, revision, device, batch_size=1,
                        dtype="bfloat16", local_files_only=False):
    from transformers import AutoTokenizer, MllamaForCausalLM
    options = {"revision": revision, "local_files_only": local_files_only}
    tokenizer = AutoTokenizer.from_pretrained(model_path, **options)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = MllamaForCausalLM.from_pretrained(model_path, torch_dtype=getattr(torch, dtype), **options)
    model.to(device).eval().requires_grad_(False)
    features = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Llama text features"):
        inputs = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                           max_length=256, return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        features.append(mean_pool(outputs.hidden_states[-1], inputs["attention_mask"]).float().cpu())
    resolved = getattr(model.config, "_commit_hash", None)
    del outputs, inputs, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(features), resolved


def image_inventory(image_dir, node_ids, extension):
    digest = hashlib.sha256()
    paths = []
    for node in node_ids.tolist():
        path = Path(image_dir) / "{}{}".format(node, extension)
        if not path.is_file():
            raise FileNotFoundError("Missing image for node {}: {}".format(node, path))
        digest.update("{}:{}\n".format(node, file_sha256(path)).encode())
        paths.append(path)
    return paths, digest.hexdigest()


@torch.no_grad()
def extract_patch_features(data, image_dir, model_path, output, revision="main",
                           device="cuda", batch_size=32, extension=".jpg", local_files_only=False):
    from PIL import Image
    ids = data.train_ids.sort().values
    paths, inventory_hash = image_inventory(image_dir, ids, extension)
    expected = {
        "kind": "clip_training_patches",
        "csv_sha256": file_sha256(data.directory / "Movies.csv"),
        "graph_sha256": file_sha256(data.directory / "MoviesGraph.pt"),
        "training_ids_sha256": json_digest(ids.tolist()),
        "image_inventory_sha256": inventory_hash,
        "model_identity": model_identity(model_path, revision),
        "grid": [2, 2],
        "crop_order": "top_left,top_right,bottom_left,bottom_right",
        "preprocessing": "EXIF-corrected RGB; CLIP default preprocessing per crop",
    }
    payload = cached_payload(output, expected)
    if payload is not None:
        validate_vectors(payload["features"], "Patch features", (len(ids), 4, 512))
        if not torch.equal(payload["node_ids"], ids):
            raise ValueError("Cached patch node order differs")
        return payload
    from transformers import CLIPModel, CLIPProcessor
    options = {"revision": revision, "local_files_only": local_files_only}
    processor = CLIPProcessor.from_pretrained(model_path, **options)
    model = CLIPModel.from_pretrained(model_path, **options).to(device).eval().requires_grad_(False)
    if model.config.projection_dim != 512 or model.config.vision_config.patch_size != 16:
        raise ValueError("Expected CLIP ViT-B/16 with 512D projected image features")
    features = torch.empty(len(ids), 4, 512)
    nodes_per_batch = max(batch_size // 4, 1)
    for start in tqdm(range(0, len(ids), nodes_per_batch), desc="CLIP training patches"):
        selected_paths = paths[start:start + nodes_per_batch]
        views = []
        for path in selected_paths:
            with Image.open(path) as image:
                views.extend(patch_views(image))
        vectors = []
        for offset in range(0, len(views), batch_size):
            inputs = processor(images=views[offset:offset + batch_size], return_tensors="pt",
                               input_data_format="channels_last").to(device)
            values = model.get_image_features(**inputs)
            vectors.append(F.normalize(values.float(), dim=-1).cpu())
        features[start:start + len(selected_paths)] = torch.cat(vectors).reshape(-1, 4, 512)
    validate_vectors(features, "Patch features", (len(ids), 4, 512))
    payload = {"features": features, "node_ids": ids,
               "metadata": dict(expected, resolved_commit=getattr(model.config, "_commit_hash", None))}
    save_tensor(output, payload)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def extract_node_text_features(data, model_path, output, revision="main", device="cuda",
                               batch_size=1, dtype="bfloat16", local_files_only=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new node feature file: {}".format(output))
    texts = [row["text"] or "" for row in data.records]
    features, resolved = llama_text_features(texts, model_path, revision, device,
                                            batch_size, dtype, local_files_only)
    validate_vectors(features, "Node text features", (len(texts), 4096))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        np.save(handle, features.numpy())
    write_json(str(output) + ".json", {
        "kind": "llama_node_text", "model_identity": model_identity(model_path, revision),
        "resolved_commit": resolved, "csv_sha256": file_sha256(data.directory / "Movies.csv"),
        "pooling": "last_hidden_attention_masked_mean", "max_length": 256,
        "chat_template": False, "dtype": dtype, "feature_sha256": file_sha256(output),
    })


@torch.no_grad()
def extract_node_image_features(data, image_dir, model_path, output, revision="main",
                                device="cuda", dtype="bfloat16", extension=".jpg", local_files_only=False):
    from PIL import Image, ImageOps
    from transformers import AutoProcessor, MllamaForConditionalGeneration
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new node feature file: {}".format(output))
    paths, inventory_hash = image_inventory(image_dir, torch.arange(len(data.records)), extension)
    options = {"revision": revision, "local_files_only": local_files_only}
    processor = AutoProcessor.from_pretrained(model_path, **options)
    model = MllamaForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=getattr(torch, dtype), **options).to(device).eval().requires_grad_(False)
    features = torch.empty(len(paths), 4096)
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}]}], add_generation_prompt=False)
    for node, path in enumerate(tqdm(paths, desc="Llama node image features")):
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            inputs = processor(images=image, text=prompt, add_special_tokens=False, return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        features[node] = mean_pool(outputs.hidden_states[-1], inputs["attention_mask"])[0].float().cpu()
    validate_vectors(features, "Node image features", (len(paths), 4096))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        np.save(handle, features.numpy())
    write_json(str(output) + ".json", {
        "kind": "llama_node_image", "model_identity": model_identity(model_path, revision),
        "resolved_commit": getattr(model.config, "_commit_hash", None),
        "csv_sha256": file_sha256(data.directory / "Movies.csv"),
        "image_inventory_sha256": inventory_hash, "pooling": "last_hidden_mean",
        "prompt": prompt, "dtype": dtype, "feature_sha256": file_sha256(output),
    })
