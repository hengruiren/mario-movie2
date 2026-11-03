import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stage1"))
sys.path.insert(0, str(ROOT))

from prototype_patch.assets import extract_patch_features, llama_text_features, mean_pool, patch_views
from prototype_patch.cli import parse_args
from prototype_patch.common import file_sha256, json_digest, save_tensor
from prototype_patch.losses import (build_teacher_scores, class_alignment, fine_distillation,
                                     fine_weight, patch_similarity, same_class_topk, symmetric_infonce)


class LossTests(unittest.TestCase):
    def test_patch_score_uses_both_directions_and_is_symmetric(self):
        basis = torch.eye(4)
        left = basis[[0, 0, 1, 1]].unsqueeze(0)
        right = basis.unsqueeze(0)
        self.assertAlmostEqual(float(patch_similarity(left, right)), 0.75)
        self.assertTrue(torch.equal(patch_similarity(left, right), patch_similarity(right, left)))
        self.assertEqual(float(patch_similarity(left, -torch.ones_like(left))), 0.0)

    def test_distillation_updates_both_images_and_keeps_negative_cosine_gradient(self):
        anchor = torch.tensor([[1.0, 0.0]], requires_grad=True)
        neighbor = torch.tensor([[[-0.8, 0.6]]], requires_grad=True)
        teacher = torch.tensor([[0.8]], requires_grad=True)
        loss = fine_distillation(anchor, neighbor, teacher, torch.tensor([[True]]))
        self.assertAlmostEqual(float(loss.detach()), 2.56, places=5)
        loss.backward()
        self.assertGreater(float(anchor.grad.norm()), 0)
        self.assertGreater(float(neighbor.grad.norm()), 0)
        self.assertIsNone(teacher.grad)

    def test_distillation_averages_per_anchor_and_masks_missing_neighbors(self):
        anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        neighbors = torch.tensor([[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]])
        scores = torch.zeros(2, 2)
        mask = torch.tensor([[True, True], [True, False]])
        self.assertEqual(float(fine_distillation(anchors, neighbors, scores, mask)), 1.0)

    def test_class_alignment_prefers_correct_prototype(self):
        prototypes = torch.eye(2, requires_grad=True)
        image = torch.tensor([[1.0, 0.0]], requires_grad=True)
        good = class_alignment(image, prototypes, torch.tensor([0]))
        bad = class_alignment(image, prototypes, torch.tensor([1]))
        self.assertLess(good, bad)
        bad.backward()
        self.assertIsNone(prototypes.grad)

    def test_symmetric_infonce_matches_two_cross_entropies(self):
        image, text = torch.randn(4, 8), torch.randn(4, 8)
        scale = torch.tensor(1.0)
        logits = scale.exp() * F.normalize(image, dim=-1) @ F.normalize(text, dim=-1).t()
        expected = (F.cross_entropy(logits, torch.arange(4))
                    + F.cross_entropy(logits.t(), torch.arange(4))) / 2
        self.assertTrue(torch.equal(symmetric_infonce(image, text, scale), expected))

    def test_fine_weight_schedule(self):
        values = [fine_weight(epoch, 0.1, 5, 5) for epoch in range(20)]
        self.assertEqual(values[:5], [0.0] * 5)
        np.testing.assert_allclose(values[5:10], [0.02, 0.04, 0.06, 0.08, 0.1])
        self.assertEqual(values[10:], [0.1] * 10)


class RetrievalTests(unittest.TestCase):
    def test_training_only_same_class_no_self_and_small_class(self):
        features = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 1.], [0., 1.], [1., 0.]])
        labels = torch.tensor([0, 0, 0, 1, 2, 0])
        pairs = same_class_topk(features, labels, torch.tensor([4, 3, 2, 1, 0]), k=5)
        self.assertEqual(pairs["anchor_ids"].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(pairs["neighbor_ids"][0, :2].tolist(), [1, 2])
        for row, anchor in enumerate(pairs["anchor_ids"]):
            ids = pairs["neighbor_ids"][row][pairs["valid"][row]]
            self.assertNotIn(int(anchor), ids.tolist())
            self.assertNotIn(5, ids.tolist())
            self.assertTrue((labels[ids] == labels[anchor]).all())
        self.assertFalse(pairs["valid"][3:].any())

    def test_teacher_covers_selected_pairs_and_rejects_missing_cache_rows(self):
        features = F.normalize(torch.randn(4, 8), dim=-1)
        pairs = same_class_topk(features, torch.tensor([0, 0, 0, 1]), torch.arange(4), k=2)
        patches = F.normalize(torch.randn(4, 4, 8), dim=-1)
        scores = build_teacher_scores(pairs, torch.arange(4), patches)
        self.assertEqual(scores.shape, (4, 2))
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())
        self.assertTrue((scores[~pairs["valid"]] == 0).all())
        with self.assertRaises(ValueError):
            build_teacher_scores(pairs, torch.arange(3), patches[:3])


class AssetAndOptionTests(unittest.TestCase):
    def test_masked_pooling_excludes_padding(self):
        hidden = torch.tensor([[[1., 2.], [3., 4.], [100., 100.]]])
        self.assertTrue(torch.equal(mean_pool(hidden, torch.tensor([[1, 1, 0]])), torch.tensor([[2., 3.]])))

    def test_four_crops_have_expected_order_and_cover_odd_image(self):
        from PIL import Image
        pixels = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
        crops = patch_views(Image.fromarray(pixels))
        self.assertEqual([crop.size for crop in crops], [(3, 2), (4, 2), (3, 3), (4, 3)])
        self.assertTrue(np.array_equal(np.asarray(crops[0]), pixels[:2, :3]))
        self.assertTrue(np.array_equal(np.asarray(crops[3]), pixels[2:, 3:]))

    def options(self, extra=()):
        return parse_args(["--data-root", "/data", "--text-feature", "/text.npy",
                           "--image-feature", "/image.npy", "--class-cache", "/classes.pt",
                           "--patch-cache", "/patches.pt", "--output-dir", "/run"] + list(extra))

    def test_defaults_and_ablation_overrides(self):
        _, settings = self.options()
        self.assertEqual(settings["lambda_class"], 1.0)
        self.assertEqual(settings["lambda_fine"], 0.1)
        self.assertEqual(settings["base_epochs"], 80)
        self.assertEqual(settings["finetune_epochs"], 20)
        _, settings = self.options(["--method", "infonce_only"])
        self.assertEqual((settings["lambda_class"], settings["lambda_fine"]), (0, 0))
        _, settings = self.options(["--lambda-fine", "0.03"])
        self.assertEqual(settings["lambda_fine"], 0.03)

    def test_invalid_weights_and_zero_temperature_are_rejected(self):
        for options in (["--lambda-fine", "-1"], ["--class-temperature", "0"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.options(options)


class FrozenModelInterfaceTests(unittest.TestCase):
    def test_llama_text_forward_with_real_small_mllama_and_padding(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import MllamaForCausalLM, PreTrainedTokenizerFast
        from transformers.models.mllama.configuration_mllama import MllamaTextConfig
        backend = Tokenizer(WordLevel({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3,
                                       "milk": 4, "and": 5, "bread": 6}, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
                                            bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]",
                                            model_input_names=["input_ids", "attention_mask"])
        config = MllamaTextConfig(vocab_size=16, hidden_size=32, intermediate_size=64,
                                  num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                                  cross_attention_layers=[], max_position_embeddings=512,
                                  bos_token_id=1, eos_token_id=2, pad_token_id=0,
                                  rope_scaling={"rope_type": "default"})
        model = MllamaForCausalLM(config).eval()
        texts = ["milk and bread", "milk"]
        with torch.no_grad():
            tokens = tokenizer(texts, padding=True, return_tensors="pt")
            expected = mean_pool(model(**tokens, output_hidden_states=True).hidden_states[-1], tokens.attention_mask)
        with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                mock.patch("transformers.MllamaForCausalLM.from_pretrained", return_value=model):
            features, _ = llama_text_features(texts, "local-test", "main", "cpu", batch_size=2, dtype="float32")
        torch.testing.assert_close(features, expected)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))

    def test_clip_extraction_with_real_small_clip_and_image_cache_validation(self):
        from PIL import Image
        from types import SimpleNamespace
        from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel, CLIPTextConfig, CLIPVisionConfig
        config = CLIPConfig.from_text_vision_configs(
            CLIPTextConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                           num_hidden_layers=1, num_attention_heads=4),
            CLIPVisionConfig(image_size=32, patch_size=16, hidden_size=32,
                             intermediate_size=64, num_hidden_layers=1, num_attention_heads=4),
            projection_dim=512)
        model = CLIPModel(config)
        processor = CLIPImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32})
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "Movies.csv").write_text("data")
            (directory / "MoviesGraph.pt").write_bytes(b"graph")
            images = directory / "images"
            images.mkdir()
            for node in (0, 2, 4):
                Image.new("RGB", (9, 7), (node * 30, 50, 120)).save(images / "{}.jpg".format(node))
            data = SimpleNamespace(directory=directory, train_ids=torch.tensor([0, 2, 4]))
            output = directory / "patches.pt"
            with mock.patch("transformers.CLIPProcessor.from_pretrained", return_value=processor), \
                    mock.patch("transformers.CLIPModel.from_pretrained", return_value=model):
                payload = extract_patch_features(data, images, "local-test", output, device="cpu", batch_size=8)
            self.assertEqual(payload["features"].shape, (3, 4, 512))
            torch.testing.assert_close(payload["features"].norm(dim=-1), torch.ones(3, 4))
            self.assertEqual(payload["node_ids"].tolist(), [0, 2, 4])
            self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
            Image.new("RGB", (9, 7), "red").save(images / "2.jpg")
            with self.assertRaises(ValueError):
                extract_patch_features(data, images, "local-test", output, device="cpu")


class OfficialEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import dgl
        from prototype_patch.training import build_official_model
        torch.set_num_threads(1)
        cls.settings = json.loads((ROOT / "configs/prototype_patch.json").read_text())
        cls.model = build_official_model(cls.settings, "cpu")
        cls.dgl = dgl

    def test_cpu_only_dgl_is_rejected_before_baseline_training(self):
        from prototype_patch.training import validate_dgl_device
        with mock.patch.object(self.dgl, "graph") as graph:
            graph.return_value.to.side_effect = self.dgl.DGLError("Device API cuda is not enabled")
            validate_dgl_device("cpu")
            graph.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "CUDA-enabled DGL"):
                validate_dgl_device("cuda:0")

    def test_one_layer_forward_is_independent_of_extra_neighbor_rows(self):
        model = self.model.eval()
        image = torch.randn(6, 1, 4096)
        text = torch.randn(6, 1, 4096)
        with torch.no_grad():
            full_text, full_image = model(None, text, torch.ones(6, 1), image)
            direct_text = model.infer(None, text[:2], torch.ones(2, 1))
            direct_image = model.image_infer(None, image[:2])
        torch.testing.assert_close(full_text[:2], direct_text)
        torch.testing.assert_close(full_image[:2], direct_image)

    def test_joint_step_changes_image_only(self):
        from types import SimpleNamespace
        from prototype_patch.common import state_digest
        from prototype_patch.training import construct_prototypes, finetune_batch, freeze_text
        model = self.model
        freeze_text(model)
        text_before = state_digest(model.bert)
        scale_before = model.logit_scale.detach().clone()
        image_before = model.image_bert.input_proj.weight.detach().clone()
        prototypes = construct_prototypes(model, torch.randn(20, 4096), "cpu")
        pairs = {"anchor_ids": torch.arange(4), "neighbor_ids": torch.tensor([[1], [0], [3], [2]]),
                 "valid": torch.ones(4, 1, dtype=torch.bool)}
        data = SimpleNamespace(image=torch.randn(4, 4096), labels=torch.tensor([0, 0, 1, 1]))
        frozen_text = torch.randn(4, 512)
        optimizer = torch.optim.Adam(model.image_bert.parameters(), lr=1e-4)
        model.image_bert.train()
        loss, record = finetune_batch(model, data, frozen_text, prototypes, pairs,
                                       torch.full((4, 1), 0.5), torch.arange(4), self.settings, 0.1, "cpu")
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(record["total"], record["infonce"] + record["class"] + 0.1 * record["fine"], places=5)
        self.assertEqual(state_digest(model.bert), text_before)
        self.assertTrue(torch.equal(model.logit_scale, scale_before))
        self.assertFalse(torch.equal(model.image_bert.input_proj.weight, image_before))
        self.assertTrue(all(parameter.grad is None for parameter in model.bert.parameters()))
        self.assertFalse(model.bert.training)
        self.assertFalse(prototypes.requires_grad)


class EndToEndTests(unittest.TestCase):
    def test_real_official_baseline_finetuning_export_and_stage2_loader(self):
        import csv
        import dgl
        from stage2.mario_llm_stage2.subset import make_manifest, save_manifest
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "MAGB/Movies"
            (directory / "TextFeature").mkdir(parents=True)
            (directory / "ImageFeature").mkdir()
            descriptions = json.loads((ROOT / "configs/movies_classes.json").read_text())
            labels = torch.arange(100) % 20
            with (directory / "Movies.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "label", "second_category", "text"])
                writer.writeheader()
                for node, label in enumerate(labels):
                    writer.writerow({"id": node, "label": int(label),
                                     "second_category": descriptions[int(label)]["name"], "text": "sample"})
            graph = dgl.graph((torch.arange(99), torch.arange(1, 100)), num_nodes=100)
            graph.ndata["label"] = labels
            dgl.save_graphs(str(directory / "MoviesGraph.pt"), [graph])
            manifest_path = root / "targets.json"
            manifest = make_manifest(labels.numpy(), directory)
            save_manifest(manifest_path, manifest)
            train_ids = torch.tensor(sorted(manifest["splits"]["train"]))
            image_path = directory / "ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy"
            text_path = directory / "TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy"
            rng = np.random.RandomState(123)
            np.save(image_path, rng.randn(100, 4096).astype(np.float32))
            np.save(text_path, rng.randn(100, 4096).astype(np.float32))
            class_cache = root / "classes.pt"
            patch_cache = root / "patches.pt"
            save_tensor(class_cache, {
                "features": torch.randn(20, 4096), "class_ids": torch.arange(20),
                "class_names": [row["name"] for row in descriptions],
                "texts": ["{}. {}".format(row["name"], row["description"]) for row in descriptions],
                "metadata": {"kind": "llama_class_text", "csv_sha256": file_sha256(directory / "Movies.csv"),
                             "descriptions_sha256": file_sha256(ROOT / "configs/movies_classes.json"),
                             "pooling": "last_hidden_attention_masked_mean", "max_length": 256, "chat_template": False}})
            save_tensor(patch_cache, {"features": F.normalize(torch.randn(60, 4, 512), dim=-1),
                "node_ids": train_ids, "metadata": {"kind": "clip_training_patches", "grid": [2, 2],
                "csv_sha256": file_sha256(directory / "Movies.csv"),
                "graph_sha256": file_sha256(directory / "MoviesGraph.pt"),
                "training_ids_sha256": json_digest(train_ids.tolist())}})
            settings = json.loads((ROOT / "configs/prototype_patch.json").read_text())
            settings.update(base_epochs=1, base_batch_size=20, finetune_epochs=2,
                            finetune_batch_size=20, class_only_epochs=0, fine_ramp_epochs=1)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(settings))
            output = root / "run"
            command = [sys.executable, str(ROOT / "stage1/train_prototype_patch.py"),
                       "--data-root", str(directory.parent), "--movies-subset", str(manifest_path),
                       "--text-feature", str(text_path), "--image-feature", str(image_path),
                       "--class-cache", str(class_cache), "--patch-cache", str(patch_cache),
                       "--output-dir", str(output), "--config", str(config_path), "--device", "cpu"]
            import os
            environment = dict(os.environ, DGLBACKEND="pytorch", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
            final = torch.load(output / "Movies_stage1_mlp.pth", weights_only=True)
            baseline = torch.load(output / "base/stage1/Movies_stage1_mlp.pth", weights_only=True)
            self.assertEqual(final.shape, (100, 2, 512))
            self.assertTrue(torch.isfinite(final).all())
            self.assertTrue(torch.equal(final[:, 0], baseline[:, 0]))
            self.assertFalse(torch.equal(final[:, 1], baseline[:, 1]))
            teacher = torch.load(output / "fixed_patch_teacher.pt", weights_only=True)
            self.assertEqual(teacher["anchor_ids"].tolist(), train_ids.tolist())
            for row, anchor in enumerate(train_ids):
                selected = teacher["neighbor_ids"][row][teacher["valid"][row]]
                self.assertTrue((labels[selected] == labels[anchor]).all())
                self.assertNotIn(int(anchor), selected.tolist())
            metrics = json.loads((output / "stage1_metrics.json").read_text())
            self.assertTrue(metrics["text_export_bitwise_unchanged"])
            self.assertEqual(metrics["final"]["nodes"], 20)
            history = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
            self.assertEqual([row["lambda_fine"] for row in history], [0.1, 0.1])
            from types import SimpleNamespace
            from stage2.mario_llm_stage2.data import load_movies
            args = SimpleNamespace(data_root=str(directory.parent), dataset="Movies", seed=42,
                                   movies_subset=str(manifest_path), stage1_feature=str(output / "Movies_stage1_mlp.pth"))
            data = load_movies(args)
            self.assertEqual((data.train_idx.numel(), data.train_target_idx.numel(),
                              data.val_idx.numel(), data.test_idx.numel()), (60, 12, 20, 20))
            # Feed actual fine-tuned embeddings and their own frozen prototypes
            # through the interstage export, then the unchanged Stage-2 loader.
            clip_all = root / "clip_all.pt"
            save_tensor(clip_all, {
                "global_features": F.normalize(torch.randn(100, 512), dim=-1),
                "patches": F.normalize(torch.randn(100, 4, 512), dim=-1),
                "node_ids": torch.arange(100),
                "metadata": {"kind": "clip_all_nodes_global_and_2x2", "grid": [2, 2],
                    "csv_sha256": file_sha256(directory / "Movies.csv"),
                    "graph_sha256": file_sha256(directory / "MoviesGraph.pt"),
                    "node_ids_sha256": json_digest(list(range(100)))}})
            refined_dir = root / "clip_class"
            refine_command = [sys.executable, str(ROOT / "scripts/class_guided_patch.py"),
                "--mode", "export", "--data-root", str(directory.parent),
                "--movies-subset", str(manifest_path),
                "--stage1-feature", str(output / "Movies_stage1_mlp.pth"),
                "--class-prototypes", str(output / "class_prototypes.pt"),
                "--clip-cache", str(clip_all), "--patch-query", "class",
                "--output-dir", str(refined_dir), "--device", "cpu"]
            refined_run = subprocess.run(refine_command, cwd=ROOT, env=environment,
                capture_output=True, text=True, timeout=60)
            self.assertEqual(refined_run.returncode, 0,
                refined_run.stdout[-3000:] + refined_run.stderr[-3000:])
            args.stage1_feature = str(refined_dir / "Movies_stage1_mlp.pth")
            refined_data = load_movies(args)
            self.assertTrue(torch.equal(refined_data.features[:, 0], final[:, 0]))
            self.assertEqual(tuple(refined_data.features.shape), (100, 2, 512))
            self.assertEqual((len(refined_data.train_idx), len(refined_data.train_target_idx),
                len(refined_data.val_idx), len(refined_data.test_idx)), (60, 12, 20, 20))


if __name__ == "__main__":
    unittest.main()
