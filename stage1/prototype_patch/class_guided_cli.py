"""Prepare, apply, and audit the optional class-guided patch experiment."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .class_guided import fit_clip_to_mario, map_clip_features, refine_consensus_batch
from .class_guided_assets import extract_all_clip, validate_clip_cache
from .common import ROOT, file_sha256, json_digest, save_tensor, validate_vectors, write_json
from .data import load_data
from .evaluation import embedding_metrics


def parse_args(argv=None):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(ROOT / "configs/class_guided_patch.json"))
    known, _ = config_parser.parse_known_args(argv)
    settings = json.loads(Path(known.config).read_text())
    allowed = {"bridge_ridge", "patch_temperature", "min_class_gain", "batch_size",
               "clip_batch_size", "probe_ridge"}
    if set(settings) != allowed:
        config_parser.error("Class-guided configuration keys must be {}".format(sorted(allowed)))
    parser = argparse.ArgumentParser(description="Class-consensus gated Mario image patch refinement")
    parser.add_argument("--config", default=known.config)
    parser.add_argument("--mode", choices=("all", "prepare", "export"), default="all")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--movies-subset", required=True)
    parser.add_argument("--stage1-feature", default="")
    parser.add_argument("--class-prototypes", default="")
    parser.add_argument("--patch-query", choices=("class", "text"), default="class",
                        help="Patch weighting query; agreement and acceptance always use original class predictions")
    parser.add_argument("--bridge-checkpoint", default="",
                        help="Reuse a frozen bridge with matching source/cache/train-split/ridge provenance")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clip-cache", default="")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch16")
    parser.add_argument("--clip-revision", default="57c216476eefef5ab752ec549e440a49ae4ae5f3")
    parser.add_argument("--image-dir", default="")
    parser.add_argument("--image-extension", default=".jpg")
    parser.add_argument("--device", default="cuda", help="Device for frozen CLIP extraction; bridge/export use CPU")
    parser.add_argument("--local-files-only", action="store_true")
    for key in sorted(allowed):
        parser.add_argument("--" + key.replace("_", "-"), type=int if key.endswith("batch_size") else float,
                            default=settings[key])
    args = parser.parse_args(argv)
    if args.mode != "prepare" and (not args.stage1_feature or not args.class_prototypes):
        parser.error("Export requires --stage1-feature and --class-prototypes from the same Stage-1 run")
    if args.batch_size < 1 or args.clip_batch_size < 1:
        parser.error("Batch sizes must be positive")
    if (any(not math.isfinite(getattr(args, k)) for k in allowed if not k.endswith("batch_size"))
            or min(args.bridge_ridge, args.patch_temperature, args.probe_ridge) <= 0
            or args.min_class_gain < 0):
        parser.error("Ridges/temperature must be finite and positive; minimum gain must be nonnegative")
    if not args.clip_cache:
        args.clip_cache = str(Path(args.output_dir) / "clip_all_nodes.pt")
    return args


def _reason(diagnostics, index, minimum_gain):
    d = diagnostics
    if not bool(d["agreement"][index]):
        return "modality_disagreement"
    if not bool(d["candidate_valid"][index]):
        return "invalid_mapped_candidate"
    if bool(d["accepted"][index]):
        return "accepted"
    class_ok = float(d["class_after"][index]) > float(d["class_before"][index]) + minimum_gain
    text_ok = float(d["text_after"][index]) >= float(d["text_before"][index])
    return "both_checks_failed" if not class_ok and not text_ok else (
        "class_not_improved" if not class_ok else "text_similarity_decreased")


def _validate_inputs(args, data):
    features = torch.load(args.stage1_feature, map_location="cpu", weights_only=True)
    if not isinstance(features, torch.Tensor) or features.shape != (len(data.records), 2, 512):
        raise ValueError("Source Stage-1 export must be a tensor [N,2,512]")
    validate_vectors(features, "Stage-1 embeddings", features.shape)
    payload = torch.load(args.class_prototypes, map_location="cpu", weights_only=True)
    prototypes = validate_vectors(payload["prototypes"], "Class prototypes", (20, 512))
    if payload.get("class_names") != data.names:
        raise ValueError("Prototype category order differs from Movies category order")
    # The original export records the frozen Text encoder's identity.
    sidecar_path = Path(str(args.stage1_feature) + ".json")
    if not sidecar_path.is_file():
        raise ValueError("Source Stage-1 provenance sidecar is required: {}".format(sidecar_path))
    sidecar = json.loads(sidecar_path.read_text())
    if sidecar.get("feature_sha256") != file_sha256(args.stage1_feature):
        raise ValueError("Source feature hash differs from its provenance sidecar")
    if (not payload.get("text_encoder_sha256") or
            payload["text_encoder_sha256"] != sidecar.get("frozen_text_sha256")):
        raise ValueError("Class prototypes and source features use different frozen Text encoders")
    signature = sidecar.get("signature", {})
    for name, filename in [("csv_sha256", "Movies.csv"), ("graph_sha256", "MoviesGraph.pt")]:
        if signature.get(name) != file_sha256(data.directory / filename):
            raise ValueError("Source feature dataset provenance differs: {}".format(filename))
    if signature.get("stage2_manifest_sha256") != file_sha256(args.movies_subset):
        raise ValueError("Source features and refinement must use the same split/target manifest")
    if payload.get("class_cache_sha256") != signature.get("class_cache_sha256"):
        raise ValueError("Prototype class cache differs from source features")
    return features, prototypes, payload, sidecar


def _reuse_bridge(path, record):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    for key in ("source_feature_sha256", "clip_cache_sha256", "bridge_training_ids_sha256",
                "bridge_training_nodes", "bridge_scope", "bridge_objective"):
        if metadata.get(key) != record[key]:
            raise ValueError("Reused bridge provenance differs: {}".format(key))
    if metadata.get("bridge_uses_labels") is not False:
        raise ValueError("Reused bridge must have training-only, label-free provenance")
    if metadata.get("inputs", {}).get("bridge_ridge") != record["inputs"]["bridge_ridge"]:
        raise ValueError("Reused bridge ridge differs")
    bridge = {}
    for key, shape in (("weight", (512, 512)), ("input_mean", (512,)), ("target_mean", (512,))):
        value = payload.get(key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError("Invalid reused bridge tensor: {}".format(key))
        bridge[key] = value
    return bridge


def export_refined(args, data):
    output = Path(args.output_dir).resolve()
    feature_path = output / "Movies_stage1_mlp.pth"
    if feature_path.exists() or (output / "run_config.json").exists():
        raise FileExistsError("Choose a fresh refinement output directory: {}".format(output))
    source, prototypes, prototype_payload, source_sidecar = _validate_inputs(args, data)
    cache = torch.load(args.clip_cache, map_location="cpu", weights_only=True)
    validate_clip_cache(cache, data)
    output.mkdir(parents=True, exist_ok=True)
    source_files = [ROOT / "stage1/prototype_patch" / name for name in
                    ("class_guided.py", "class_guided_assets.py", "class_guided_cli.py")]
    record = dict(method="class_consensus_patch_refinement" if args.patch_query == "class"
                         else "text_guided_consensus_patch_refinement", inputs=vars(args),
                  source_feature_sha256=file_sha256(args.stage1_feature),
                  source_prototypes_sha256=file_sha256(args.class_prototypes),
                  clip_cache_sha256=file_sha256(args.clip_cache), clip_metadata=cache["metadata"],
                  movies_subset_sha256=file_sha256(args.movies_subset),
                  bridge_training_ids_sha256=json_digest(data.train_ids.tolist()),
                  bridge_training_nodes=len(data.train_ids), bridge_uses_labels=False,
                  bridge_scope="full training split only; normalized CLIP global -> normalized Mario image",
                  bridge_objective="sum squared residuals + ridge * squared weight norm; unregularized intercept",
                  decision_uses_labels=False, evaluation_labels="validation only",
                  disagreement_policy="exact original image; no weighting",
                  acceptance="class_after > class_before + min_class_gain AND text_after >= text_before",
                  patch_space="CLIP crops -> training-fitted affine bridge -> unit Mario coordinates",
                  text_unchanged=True, accepted_image_norm="preserve original norm",
                  class_margin_is_not_an_acceptance_gate=True,
                  patch_weighting_query="own_text_embedding" if args.patch_query == "text"
                                       else "shared_predicted_class_prototype",
                  bridge_checkpoint_sha256=file_sha256(args.bridge_checkpoint) if args.bridge_checkpoint else None,
                  source_sha256={str(p.relative_to(ROOT)): file_sha256(p) for p in source_files})
    bridge = _reuse_bridge(args.bridge_checkpoint, record) if args.bridge_checkpoint else None
    write_json(output / "run_config.json", record)
    if bridge is None:
        print("Fitting CLIP -> Mario bridge on {} TRAINING full images only".format(len(data.train_ids)), flush=True)
        bridge = fit_clip_to_mario(cache["global_features"], source[:, 1], data.train_ids, args.bridge_ridge)
    else:
        print("Reusing frozen CLIP -> Mario bridge: {}".format(args.bridge_checkpoint), flush=True)
    save_tensor(output / "clip_to_mario_bridge.pt", dict(bridge, metadata=record))
    refined = source.clone()
    totals = Counter()
    split_stats = {name: Counter() for name in ("train", "validation", "test")}
    membership = {int(u): name for name, ids in (("train", data.train_ids),
                  ("validation", data.val_ids), ("test", data.test_ids)) for u in ids}
    with (output / "node_decisions.jsonl").open("w") as handle:
        for start in tqdm(range(0, len(source), args.batch_size), desc="Class-consensus patch refinement"):
            stop = min(start + args.batch_size, len(source))
            mapped = map_clip_features(cache["patches"][start:stop], bridge)
            image, diagnostics = refine_consensus_batch(source[start:stop, 0], source[start:stop, 1],
                                  mapped, prototypes, args.patch_temperature, args.min_class_gain, args.patch_query)
            refined[start:stop, 1] = image
            for i, u in enumerate(range(start, stop)):
                reason = _reason(diagnostics, i, args.min_class_gain)
                totals[reason] += 1
                split_stats[membership[u]][reason] += 1
                decision = dict(node=u, split=membership[u], reason=reason,
                                text_class=int(diagnostics["text_class"][i]),
                                image_class=int(diagnostics["image_class"][i]),
                                agreement=bool(diagnostics["agreement"][i]),
                                accepted=bool(diagnostics["accepted"][i]))
                if decision["agreement"]:
                    decision.update(candidate_class=int(diagnostics["candidate_class"][i]),
                                    patch_weights=diagnostics["patch_weights"][i].tolist())
                    for key in ("class_before", "class_after", "text_before", "text_after"):
                        decision[key] = float(diagnostics[key][i])
                handle.write(json.dumps(decision, sort_keys=True, allow_nan=False) + "\n")
    assert torch.equal(source[:, 0], refined[:, 0])
    validate_vectors(refined, "Refined embeddings", source.shape)
    save_tensor(feature_path, refined)
    save_tensor(output / "class_prototypes.pt", prototype_payload)
    print("Evaluating BEFORE/AFTER on the full validation split only", flush=True)
    metrics = dict(before=embedding_metrics(source[:, 1], source[:, 0], data, prototypes,
                                           args.probe_ridge, args.batch_size),
                   after=embedding_metrics(refined[:, 1], source[:, 0], data, prototypes,
                                          args.probe_ridge, args.batch_size))
    bridge_quality = {}
    for name, ids in (("train", data.train_ids), ("validation", data.val_ids)):
        mapped = map_clip_features(cache["global_features"][ids], bridge)
        bridge_quality[name + "_global_cosine_mean"] = float(
            (mapped * F.normalize(source[ids, 1].float(), dim=-1)).sum(-1).mean())
    metrics.update(evaluation_split="full_validation", node_decision_counts=dict(totals),
                   split_decision_counts={k: dict(v) for k, v in split_stats.items()},
                   bridge_quality=bridge_quality, text_export_bitwise_unchanged=True,
                   nodes=len(source), accepted_nodes=totals["accepted"],
                   decision_uses_labels=False, bridge_training_nodes=len(data.train_ids))
    write_json(output / "refinement_metrics.json", metrics)
    write_json(str(feature_path) + ".json", dict(method=record["method"], shape=list(refined.shape),
               feature_sha256=file_sha256(feature_path), source_feature_sha256=record["source_feature_sha256"],
               run_config_sha256=file_sha256(output / "run_config.json"),
               frozen_text_sha256=source_sidecar["frozen_text_sha256"],
               signature=source_sidecar["signature"]))
    print(json.dumps(dict(stage2_feature=str(feature_path), metrics=metrics), indent=2), flush=True)
    return feature_path


def main(argv=None):
    args = parse_args(argv)
    data = load_data(args.data_root, args.movies_subset)
    if args.mode != "prepare":
        output = Path(args.output_dir)
        if (output / "Movies_stage1_mlp.pth").exists() or (output / "run_config.json").exists():
            raise FileExistsError("Choose a fresh refinement output directory: {}".format(output))
        _validate_inputs(args, data)
    if args.mode in ("all", "prepare"):
        image_dir = args.image_dir or str(data.directory / "MoviesImages")
        if not args.image_dir and not Path(image_dir).is_dir():
            image_dir = str(data.directory / "MoviesImages")
        extract_all_clip(data, image_dir, args.clip_model, args.clip_cache, args.clip_revision,
                         args.device, args.clip_batch_size, args.image_extension, args.local_files_only)
    if args.mode in ("all", "export"):
        export_refined(args, data)


if __name__ == "__main__":
    main()
