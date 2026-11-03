import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .common import (ROOT, file_sha256, json_digest, save_tensor, seed_everything,
                     state_digest, validate_vectors, working_directory, write_json)
from .data import data_signature, load_data, load_descriptions
from .evaluation import embedding_metrics, encode_images, prototype_metrics
from .losses import (build_teacher_scores, class_alignment, fine_distillation,
                     fine_weight, same_class_topk, symmetric_infonce)


def original_module():
    name = "mario_official_stage1"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "stage1/main.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def official_config(settings, checkpoint_folder):
    return SimpleNamespace(
        mode="gpu", n_epochs=settings["base_epochs"], dataset="Movies",
        checkpoint_folder=str(checkpoint_folder), batch_size=settings["base_batch_size"],
        wd=settings["weight_decay"], lr=settings["base_lr"], full_neighbor=False,
        num_of_neighbors=settings["num_neighbors"], num_layers=settings["num_layers"],
        lr_scheduler_step_size=settings["base_scheduler_step"],
        lr_scheduler_gamma=settings["base_scheduler_gamma"], patch_size=5,
        hidden_dim=settings["hidden_dim"], seed=settings["seed"], threshold=0.0,
        filter=False, eval_gnn_layers=1, save=False, id=0, num=0, use_large_features=True)


def build_official_model(settings, device):
    cfg = official_config(settings, "check")
    from src.models.tnlrv3.configuration_tnlrv3 import TuringNLRv3Config
    from src.models.modeling_graphformers import GraphFormersForNeighborPredict
    config = TuringNLRv3Config.from_pretrained(str(ROOT / "stage1/config.json"), output_hidden_states=True)
    config.hidden_size = settings["hidden_dim"]
    config.num_hidden_layers = 1
    return GraphFormersForNeighborPredict(config, cfg).to(device)


def validate_dgl_device(device):
    device = torch.device(device)
    if device.type != "cuda":
        return
    import dgl
    try:
        dgl.graph(([0], [0]), num_nodes=1).to(device)
    except dgl.DGLError as error:
        raise RuntimeError(
            "Stage 1 requires a working CUDA-enabled DGL build to move sampled graph blocks. "
            "Install the GPU dependencies from requirements.txt "
            "and verify DGL graph.to('cuda:0') before starting training."
        ) from error


def baseline_training(args, settings, signature):
    import dgl
    from dataset_amazon_small import NodeClassificationDataset
    from scripts.prepare_movies_inputs import prepare
    original = original_module()
    dgl.seed(settings["seed"])
    work = Path(args.output_dir) / "base"
    prepare(args.data_root, work, args.text_feature, args.image_feature)
    stage1_dir = work / "stage1"
    cfg = official_config(settings, stage1_dir / "check")
    device = torch.device(args.device)
    dataset = NodeClassificationDataset(root=str(work / "data/Movies"),
                                        data_path=str(work / "data"), verbose=True,
                                        device=device, save=False, use_large_features=True)
    graph = dgl.remove_self_loop(dataset.graph.cpu())
    graph, _ = original.to_bidirected_with_reverse_mapping(graph)
    graph = dgl.add_self_loop(graph)
    splits = {name + "_idx": graph.ndata[name + "_mask"].nonzero()
              for name in ("train", "val", "test")}
    model = build_official_model(settings, device)
    with working_directory(stage1_dir):
        original.train(cfg, graph, splits, model)
    feature_path = stage1_dir / "Movies_stage1_mlp.pth"
    embeddings = torch.load(feature_path, map_location="cpu", weights_only=True).float()
    checkpoint = {"model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                  "settings": settings, "signature": signature,
                  "feature_file": feature_path.name, "feature_sha256": file_sha256(feature_path)}
    save_tensor(stage1_dir / "encoder.pt", checkpoint)
    return model, embeddings, stage1_dir / "encoder.pt"


def load_baseline(path, settings, signature, device):
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    base_keys = ("seed", "base_epochs", "base_batch_size", "base_lr", "weight_decay",
                 "base_scheduler_step", "base_scheduler_gamma", "num_neighbors", "hidden_dim", "num_layers")
    if any(checkpoint["settings"].get(key) != settings[key] for key in base_keys):
        raise ValueError("Baseline checkpoint training settings differ")
    if checkpoint["signature"] != signature:
        raise ValueError("Baseline checkpoint dataset or feature provenance differs")
    features = path.parent / checkpoint["feature_file"]
    if file_sha256(features) != checkpoint["feature_sha256"]:
        raise ValueError("Baseline embedding file changed")
    model = build_official_model(settings, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, torch.load(features, map_location="cpu", weights_only=True).float(), path


def freeze_text(model):
    model.zero_grad(set_to_none=True)
    model.bert.requires_grad_(False).eval()
    model.logit_scale.requires_grad_(False)
    model.image_bert.requires_grad_(True)


@torch.no_grad()
def construct_prototypes(model, class_features, device):
    model.bert.eval()
    features = class_features.to(device).unsqueeze(1)
    mask = torch.ones(features.shape[:2], dtype=torch.long, device=device)
    return F.normalize(model.infer(None, features, mask), dim=-1).detach().cpu()


def finetune_batch(model, data, frozen_text, prototypes, pairs, teacher_scores,
                    rows, settings, lambda_fine, device):
    anchors = pairs["anchor_ids"][rows]
    valid = pairs["valid"][rows]
    neighbor_ids = pairs["neighbor_ids"][rows]
    safe_neighbors = torch.where(valid, neighbor_ids, anchors[:, None])
    involved = torch.cat([anchors, safe_neighbors.flatten()]) if lambda_fine > 0 else anchors
    unique_ids = involved.unique(sorted=True)
    image = model.image_infer(None, data.image[unique_ids].to(device).unsqueeze(1))
    anchor_image = image[torch.searchsorted(unique_ids, anchors).to(device)]
    text = frozen_text[anchors].to(device).detach()
    infonce = symmetric_infonce(anchor_image, text, model.logit_scale.detach())
    class_loss = class_alignment(anchor_image, prototypes.to(device), data.labels[anchors].to(device),
                                  settings["class_temperature"])
    fine_loss = anchor_image.sum() * 0.0
    if lambda_fine > 0:
        neighbor_image = image[torch.searchsorted(unique_ids, safe_neighbors).to(device)]
        fine_loss = fine_distillation(anchor_image, neighbor_image,
                                       teacher_scores[rows].to(device), valid.to(device))
    loss = infonce + settings["lambda_class"] * class_loss + lambda_fine * fine_loss
    return loss, {"infonce": float(infonce.detach()), "class": float(class_loss.detach()),
                  "fine": float(fine_loss.detach()), "total": float(loss.detach())}


def load_assets(args, data):
    classes = torch.load(args.class_cache, map_location="cpu", weights_only=True)
    patches = torch.load(args.patch_cache, map_location="cpu", weights_only=True)
    descriptions = load_descriptions(args.class_descriptions, data.names)
    metadata = classes.get("metadata", {})
    if (metadata.get("kind") != "llama_class_text"
            or metadata.get("csv_sha256") != file_sha256(data.directory / "Movies.csv")
            or metadata.get("descriptions_sha256") != file_sha256(args.class_descriptions)
            or metadata.get("pooling") != "last_hidden_attention_masked_mean"
            or metadata.get("max_length") != 256 or metadata.get("chat_template") is not False
            or classes.get("class_names") != data.names
            or classes.get("texts") != ["{}. {}".format(row["name"], row["description"]) for row in descriptions]
            or not torch.equal(classes["class_ids"], torch.arange(20))):
        raise ValueError("Class cache metadata, class IDs or descriptions differ")
    classes["features"] = validate_vectors(classes["features"], "Class features", (20, 4096))
    metadata = patches.get("metadata", {})
    if (metadata.get("kind") != "clip_training_patches" or metadata.get("grid") != [2, 2]
            or metadata.get("csv_sha256") != file_sha256(data.directory / "Movies.csv")
            or metadata.get("graph_sha256") != file_sha256(data.directory / "MoviesGraph.pt")
            or metadata.get("training_ids_sha256") != json_digest(data.train_ids.tolist())
            or not torch.equal(patches["node_ids"], data.train_ids)):
        raise ValueError("Patch cache does not match the full training split")
    patches["features"] = validate_vectors(patches["features"], "Patch features", (len(data.train_ids), 4, 512))
    return classes, patches


def train_finetuning(model, data, baseline, prototypes, pairs, scores, settings, output, device):
    freeze_text(model)
    frozen_digest = state_digest(model.bert)
    original_scale = model.logit_scale.detach().clone()
    frozen_text = baseline[:, 0].clone()
    optimizer = torch.optim.Adam(model.image_bert.parameters(), lr=settings["finetune_lr"],
                                  weight_decay=settings["weight_decay"])
    best_accuracy = -1.0
    seed_everything(settings["seed"] + 1)
    for epoch in range(settings["finetune_epochs"]):
        model.bert.eval()
        model.image_bert.train()
        coefficient = fine_weight(epoch, settings["lambda_fine"], settings["class_only_epochs"],
                                  settings["fine_ramp_epochs"])
        generator = torch.Generator().manual_seed(settings["seed"] + 10000 + epoch)
        order = torch.randperm(len(data.train_ids), generator=generator)
        totals = {name: 0.0 for name in ("infonce", "class", "fine", "total")}
        seen = 0
        for start in tqdm(range(0, len(order), settings["finetune_batch_size"]),
                          desc="Image fine-tuning {}/{}".format(epoch + 1, settings["finetune_epochs"])):
            rows = order[start:start + settings["finetune_batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss, values = finetune_batch(model, data, frozen_text, prototypes, pairs, scores,
                                          rows, settings, coefficient, device)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite fine-tuning loss")
            loss.backward()
            optimizer.step()
            for key, value in values.items():
                totals[key] += value * len(rows)
            seen += len(rows)
        if state_digest(model.bert) != frozen_digest or not torch.equal(model.logit_scale, original_scale):
            raise RuntimeError("Frozen text encoder or InfoNCE temperature changed")
        val_image = encode_images(model, data.image, data.val_ids, device, settings["eval_batch_size"])
        validation = prototype_metrics(val_image, prototypes, data.labels[data.val_ids])
        phase = ("joint" if settings["lambda_class"] > 0 else "fine_only") if coefficient > 0 else (
            "class_only" if settings["lambda_class"] > 0 else "infonce_only")
        record = {"epoch": epoch + 1, "phase": phase,
                  "lambda_class": settings["lambda_class"], "lambda_fine": coefficient,
                  "train_nodes": seen, "loss": {key: value / seen for key, value in totals.items()},
                  "weighted_class_loss": settings["lambda_class"] * totals["class"] / seen,
                  "weighted_fine_loss": coefficient * totals["fine"] / seen,
                  "validation": validation}
        with (output / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)
        checkpoint = {"model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                      "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
                      "settings": settings, "frozen_text_sha256": frozen_digest,
                      "validation": validation}
        save_tensor(output / "last_encoder.pt", checkpoint)
        if validation["prototype_accuracy"] > best_accuracy:
            best_accuracy = validation["prototype_accuracy"]
            save_tensor(output / "best_prototype_encoder.pt", checkpoint)
    return frozen_digest


def run(args, settings):
    if settings["num_layers"] != 1 or settings["hidden_dim"] != 512:
        raise ValueError("Direct fine-tuning is defined only for the official one-layer 512D encoder")
    if torch.cuda.device_count() > 1:
        raise ValueError("Expose one GPU using CUDA_VISIBLE_DEVICES for Stage 1")
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("Choose a new Stage-1 output directory: {}".format(output))
    if not args.base_checkpoint:
        validate_dgl_device(args.device)
    data = load_data(args.data_root, args.movies_subset, args.text_feature, args.image_feature)
    classes, patches = load_assets(args, data)
    signature = data_signature(data, args.movies_subset, args.text_feature, args.image_feature)
    signature.update({"class_cache_sha256": file_sha256(args.class_cache),
                      "patch_cache_sha256": file_sha256(args.patch_cache)})
    output.mkdir(parents=True)
    versions = {}
    for package in ("torch", "numpy", "transformers", "dgl"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    source_files = list((ROOT / "stage1/prototype_patch").glob("*.py")) + [
        ROOT / name for name in (
            "stage1/main.py", "stage1/config.json", "stage1/dataset_amazon_small.py",
            "stage1/src/models/modeling_graphformers.py", "stage1/src/models/tnlrv3/modeling.py",
            "stage1/src/models/tnlrv3/configuration_tnlrv3.py", "scripts/prepare_movies_inputs.py",
            "stage1/train_prototype_patch.py", "requirements.txt")]
    config_record = {"method": "prototype_patch", "variant": args.method,
                     "settings": settings, "inputs": vars(args),
                     "signature": signature, "versions": versions,
                     "class_cache_metadata": classes["metadata"], "patch_cache_metadata": patches["metadata"],
                     "source_sha256": {str(path.relative_to(ROOT)): file_sha256(path)
                                       for path in sorted(source_files)},
                     "selection": "last", "text_frozen_after_base": True,
                     "logit_scale_frozen_after_base": True,
                     "fine_student_similarity": "raw_cosine_without_clipping",
                     "fine_teacher_similarity": "mean_of_bidirectional_max_clipped_cosine",
                     "retrieval": "fixed_same_class_top_k_from_baseline_image_embeddings",
                     "label_scope": "Stage 1 uses all training labels; Stage 2 uses fixed 20% training targets"}
    write_json(output / "run_config.json", config_record)
    seed_everything(settings["seed"])
    baseline_signature = {key: value for key, value in signature.items()
                          if key not in ("class_cache_sha256", "patch_cache_sha256")}
    if args.base_checkpoint:
        model, baseline, base_path = load_baseline(args.base_checkpoint, settings, baseline_signature, args.device)
    else:
        model, baseline, base_path = baseline_training(args, settings, baseline_signature)
    if baseline.shape != (len(data.labels), 2, 512) or not torch.isfinite(baseline).all():
        raise ValueError("Baseline embeddings must be finite [N, 2, 512]")
    freeze_text(model)
    config_record["frozen_infonce_logit_scale"] = float(model.logit_scale.detach().cpu())
    config_record["frozen_infonce_temperature"] = float((-model.logit_scale.detach().cpu()).exp())
    write_json(output / "run_config.json", config_record)
    prototypes = construct_prototypes(model, classes["features"], args.device)
    validate_vectors(prototypes, "Class prototypes", (20, 512))
    save_tensor(output / "class_prototypes.pt", {"prototypes": prototypes, "class_names": data.names,
                                                "class_cache_sha256": file_sha256(args.class_cache),
                                                "text_encoder_sha256": state_digest(model.bert)})
    pairs = same_class_topk(baseline[:, 1], data.labels, data.train_ids,
                            settings["top_k"], settings["eval_batch_size"])
    scores = build_teacher_scores(pairs, patches["node_ids"], patches["features"])
    save_tensor(output / "fixed_patch_teacher.pt", dict(pairs, scores=scores,
                metadata={"baseline_checkpoint_sha256": file_sha256(base_path),
                          "patch_cache_sha256": file_sha256(args.patch_cache),
                          "training_ids_sha256": json_digest(data.train_ids.tolist())}))
    before = embedding_metrics(baseline[:, 1], baseline[:, 0], data, prototypes,
                                settings["probe_ridge"], settings["eval_batch_size"])
    selected_scores = scores[pairs["valid"]]
    write_json(output / "baseline_metrics.json", before)
    write_json(output / "teacher_statistics.json", {
        "pairs": int(pairs["valid"].sum()), "anchors_without_pairs": int((~pairs["valid"].any(dim=1)).sum()),
        "score_mean": float(selected_scores.mean()) if len(selected_scores) else None,
        "score_std": float(selected_scores.std(unbiased=False)) if len(selected_scores) else None,
        "score_min": float(selected_scores.min()) if len(selected_scores) else None,
        "score_max": float(selected_scores.max()) if len(selected_scores) else None,
        "prototype_cosine": (prototypes @ prototypes.t()).tolist(),
    })
    frozen_digest = train_finetuning(model, data, baseline, prototypes, pairs, scores,
                                      settings, output, args.device)
    final_image = encode_images(model, data.image, torch.arange(len(data.labels)),
                                args.device, settings["eval_batch_size"])
    final_embeddings = torch.stack([baseline[:, 0], final_image], dim=1)
    feature_path = output / "Movies_stage1_mlp.pth"
    save_tensor(feature_path, final_embeddings)
    after = embedding_metrics(final_image, baseline[:, 0], data, prototypes,
                               settings["probe_ridge"], settings["eval_batch_size"])
    write_json(output / "stage1_metrics.json", {"baseline": before, "final": after,
                                               "evaluation_split": "full_validation",
                                               "export_selection": "last",
                                               "frozen_text_sha256": frozen_digest,
                                               "text_export_bitwise_unchanged": torch.equal(
                                                   final_embeddings[:, 0], baseline[:, 0])})
    write_json(str(feature_path) + ".json", {"method": "prototype_patch", "shape": list(final_embeddings.shape),
               "feature_sha256": file_sha256(feature_path), "run_config_sha256": file_sha256(output / "run_config.json"),
               "base_checkpoint_sha256": file_sha256(base_path), "frozen_text_sha256": frozen_digest,
               "epoch": settings["finetune_epochs"], "selection": "last", "signature": signature})
    print(json.dumps({"stage1_metrics": str(output / "stage1_metrics.json"),
                      "stage2_feature": str(feature_path)}, indent=2), flush=True)
    return feature_path
