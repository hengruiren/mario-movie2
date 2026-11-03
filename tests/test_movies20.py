import contextlib
import copy
import importlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stage2"))
from mario_llm_stage2.subset import file_sha256, load_manifest, make_manifest, save_manifest
from mario_llm_stage2 import cli


class SubsetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        (self.directory / "Movies.csv").write_text("id,label\n0,0\n")
        (self.directory / "MoviesGraph.pt").write_bytes(b"graph fingerprint")
        self.labels = np.arange(16672) % 20
        self.manifest = make_manifest(self.labels, self.directory)
        self.path = self.directory / "targets.json"
        save_manifest(self.path, self.manifest)

    def test_counts_and_split_isolation(self):
        manifest = load_manifest(self.path, "Movies", self.labels, self.directory)
        self.assertEqual([len(manifest["splits"][k]) for k in ("train", "val", "test")],
                         [10003, 3334, 3335])
        targets = set(manifest["targets"]["train"])
        self.assertEqual(len(targets), 2001)
        self.assertLessEqual(targets, set(manifest["splits"]["train"]))
        self.assertFalse(targets & set(manifest["splits"]["val"]))
        self.assertFalse(targets & set(manifest["splits"]["test"]))
        self.assertEqual(set(manifest["targets"]), {"train"})
        expected = np.random.RandomState(42).permutation(16672).tolist()
        self.assertEqual(manifest["splits"]["train"], expected[:10003])

    def test_repeat_and_rng_independence(self):
        np.random.seed(987)
        expected = np.random.rand(5)
        np.random.seed(987)
        repeated = make_manifest(self.labels, self.directory)
        self.assertEqual(self.manifest, repeated)
        np.testing.assert_array_equal(np.random.rand(5), expected)
        before = self.path.read_bytes()
        save_manifest(self.path, repeated)
        self.assertEqual(before, self.path.read_bytes())

    def test_changed_manifest_is_not_overwritten(self):
        with self.assertRaises(ValueError):
            save_manifest(self.path, make_manifest(self.labels, self.directory, 43))

    def test_changed_sources_and_labels_are_rejected(self):
        labels = self.labels.copy()
        labels[0] = 99
        with self.assertRaises(ValueError):
            load_manifest(self.path, "Movies", labels, self.directory)
        (self.directory / "MoviesGraph.pt").write_bytes(b"changed graph")
        with self.assertRaises(ValueError):
            load_manifest(self.path, "Movies", self.labels, self.directory)

    def test_invalid_targets_and_splits_are_rejected(self):
        for kind in ("test_target", "duplicate", "split_overlap", "wrong_protocol"):
            with self.subTest(kind=kind):
                bad = copy.deepcopy(self.manifest)
                if kind == "test_target":
                    bad["targets"]["train"][0] = bad["splits"]["test"][0]
                elif kind == "duplicate":
                    bad["targets"]["train"][0] = bad["targets"]["train"][1]
                elif kind == "split_overlap":
                    bad["splits"]["val"][0] = bad["splits"]["train"][0]
                else:
                    bad["protocol"] = "both_stages_20pct"
                self.path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    load_manifest(self.path, "Movies", self.labels, self.directory)

    def test_bundled_manifest_counts(self):
        manifest = json.loads((ROOT / "splits/Movies/stage2_train20_seed42.json").read_text())
        self.assertEqual(len(manifest["targets"]["train"]), 2001)
        self.assertEqual(len(set(manifest["targets"]["train"])), 2001)
        self.assertLessEqual(set(manifest["targets"]["train"]), set(manifest["splits"]["train"]))
        self.assertEqual(sum(manifest["class_counts"]["train"].values()), 2001)


class CliTests(unittest.TestCase):
    def parse(self, options):
        with mock.patch.object(sys, "argv", ["train_llm.py", "--llm-model-path", "/model"] + options):
            return cli.parse_args()

    def test_original_default_and_subset_options(self):
        self.assertEqual(self.parse([]).movies_subset, "")
        self.assertEqual(self.parse([]).epochs, 8)
        self.assertEqual(self.parse([]).patience, 2)
        args = self.parse(["--dataset", "Movies", "--movies-subset", "targets.json"])
        self.assertEqual(args.movies_subset, "targets.json")

    def test_rejects_other_dataset_and_secondary_truncation(self):
        for options in (["--dataset", "Grocery"],
                        ["--dataset", "Movies", "--limit-train-steps", "1"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.parse(["--movies-subset", "targets.json"] + options)

    def test_help_formats_percent(self):
        with contextlib.redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit) as caught:
            self.parse(["--help"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("20% training-target", output.getvalue())

    def test_original_batch_and_top10_defaults_and_invalid_limits(self):
        self.assertEqual(self.parse([]).batch_size, 3)
        self.assertEqual(self.parse([]).top_k, 10)
        for options in (["--top-k", "0"], ["--top-k", "-1"], ["--epochs", "0"], ["--batch-size", "0"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.parse(options)


class DataAndTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        directory = root / "Movies"
        directory.mkdir()
        labels = np.arange(100) % 4
        pd.DataFrame({"id": np.arange(100), "label": labels,
                      "second_category": ["class{}".format(i) for i in labels],
                      "text": ["node text"] * 100}).to_csv(directory / "Movies.csv", index=False)
        (directory / "MoviesGraph.pt").write_bytes(b"graph")
        self.manifest_path = root / "targets.json"
        save_manifest(self.manifest_path, make_manifest(labels, directory))
        self.feature_path = root / "features.pth"
        torch.save(torch.randn(100, 2, 8), self.feature_path)
        self.graph = types.SimpleNamespace(ndata={"label": torch.tensor(labels)}, num_nodes=lambda: 100)
        self.graph.cpu = lambda: self.graph
        fake_dgl = types.ModuleType("dgl")
        fake_dgl.load_graphs = lambda _: ([self.graph], {})
        fake_dgl.remove_self_loop = lambda graph: graph
        fake_dgl.to_bidirected = lambda graph, **kwargs: graph
        fake_dgl.add_self_loop = lambda graph: graph
        fake_model = types.ModuleType("mario_llm_stage2.model")
        fake_model.MarioLLM = torch.nn.Module
        fake_checkpoint = types.ModuleType("mario_llm_stage2.checkpoint")
        fake_checkpoint.load_checkpoint = mock.Mock()
        fake_checkpoint.save_checkpoint = mock.Mock()
        fake_distributed = types.ModuleType("mario_llm_stage2.distributed")
        fake_distributed.cleanup_distributed = lambda _: None
        fake_distributed.distributed_mean = lambda value, _: value
        fake_distributed.is_main_process = lambda rank: rank == 0
        fake_distributed.seed_everything = lambda _: None
        fake_distributed.setup_distributed = lambda _: (1, 0, 0)
        modules = {"dgl": fake_dgl, fake_model.__name__: fake_model,
                   fake_checkpoint.__name__: fake_checkpoint,
                   fake_distributed.__name__: fake_distributed}
        missing = object()
        previous_modules = {name: sys.modules.get(name, missing) for name in modules}
        sys.modules.update(modules)

        def restore_modules():
            for name, previous in previous_modules.items():
                if previous is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous

        self.addCleanup(restore_modules)
        for name in ("data", "pipeline", "evaluation"):
            sys.modules.pop("mario_llm_stage2." + name, None)
        self.data_module = importlib.import_module("mario_llm_stage2.data")
        self.pipeline = importlib.import_module("mario_llm_stage2.pipeline")
        self.args = types.SimpleNamespace(
            dataset="Movies", data_root=str(root), seed=42,
            movies_subset=str(self.manifest_path), stage1_feature=str(self.feature_path),
            limit_train_steps=0, batch_size=4, epochs=1, log_every=50,
            inference_policy="router", skip_test_generation=True)

    def test_loader_keeps_full_features_validation_test_and_neighbor_pool(self):
        data = self.data_module.load_movies(self.args)
        self.assertIs(data.graph, self.graph)
        self.assertEqual(data.features.shape, (100, 2, 8))
        self.assertEqual(data.train_idx.numel(), 60)
        self.assertEqual(data.train_target_idx.numel(), 12)
        self.assertEqual(data.val_idx.numel(), 20)
        self.assertEqual(data.test_idx.numel(), 20)
        self.assertEqual(data.movies_subset_sha256, file_sha256(self.manifest_path))
        self.args.seed = 99
        changed_seed = self.data_module.load_movies(self.args)
        self.assertTrue(torch.equal(data.train_idx, changed_seed.train_idx))
        self.assertTrue(torch.equal(data.train_target_idx, changed_seed.train_target_idx))
        self.assertTrue(torch.equal(data.val_idx, changed_seed.val_idx))

    def test_without_option_preserves_full_training(self):
        self.args.movies_subset = ""
        data = self.data_module.load_movies(self.args)
        self.assertIs(data.train_target_idx, data.train_idx)
        self.assertEqual(data.train_idx.numel(), 60)

    def test_model_receives_full_labeled_neighbor_pool(self):
        data = self.data_module.load_movies(self.args)
        with mock.patch.object(self.pipeline, "MarioLLM") as constructor:
            self.pipeline._build_model(data, self.args)
        self.assertIs(constructor.call_args.kwargs["train_idx"], data.train_idx)

    def test_real_training_loop_only_updates_on_selected_targets(self):
        data = self.data_module.load_movies(self.args)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))
                self.device = torch.device("cpu")
                self.last_info = {}
                self.seen = []

            def forward(self, nodes):
                self.seen.extend(nodes.tolist())
                return (self.weight - 1).square()

        model = Model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with mock.patch.object(self.pipeline, "tqdm", side_effect=lambda values, **kwargs: values):
            loss = self.pipeline._train_one_epoch(0, model, model, optimizer, data, self.args, 0, 1)
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(model.weight.item(), 0)
        self.assertEqual(sorted(model.seen), sorted(data.train_target_idx.tolist()))
        self.assertEqual(len(model.seen), 12)

    def test_early_stopping_uses_best_loss_resets_patience_and_caps_at_eight(self):
        data = self.data_module.load_movies(self.args)
        self.args.epochs = 8
        self.args.patience = 2
        self.args.val_every = 1
        self.args.limit_val_steps = 0
        cases = [
            ([0.5, 0.4, 0.41, 0.405], 4, 1, 0.4, True),
            ([0.5, 0.4, 0.41, 0.39, 0.4, 0.41], 6, 3, 0.39, True),
            ([0.5, 0.5, 0.5], 3, 0, 0.5, True),
            ([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], 8, 7, 0.2, False),
        ]
        for index, (losses, expected_seen, best_epoch, best_loss, stopped) in enumerate(cases):
            with self.subTest(losses=losses):
                output = self.feature_path.parent / ("early_stop_" + str(index))
                output.mkdir()
                state = self.pipeline.TrainingState()
                should_stop = False
                seen = 0
                with mock.patch.object(self.pipeline, "evaluate_loss", side_effect=losses), \
                        mock.patch.object(self.pipeline, "save_checkpoint") as save, \
                        contextlib.redirect_stdout(io.StringIO()):
                    for epoch in range(self.args.epochs):
                        _, _, should_stop = self.pipeline._validate_and_checkpoint(
                            epoch, None, data, self.args, output, state, 1.0)
                        seen += 1
                        if should_stop:
                            break
                self.assertEqual(seen, expected_seen)
                self.assertEqual(should_stop, stopped)
                self.assertEqual(state.best_epoch, best_epoch)
                self.assertAlmostEqual(state.best_val, best_loss)
                self.assertEqual(save.call_args.args[2], best_epoch)
                self.assertEqual(state.best_checkpoint_path, output / "best_stage2_llm.pt")

    def test_metrics_report_target_and_neighbor_pool_separately(self):
        data = self.data_module.load_movies(self.args)
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = self.pipeline._write_metrics_and_test(
                None, data, self.args, self.feature_path.parent, self.pipeline.TrainingState(), 1)
        self.assertEqual(metrics["train_target_nodes"], 12)
        self.assertEqual(metrics["train_neighbor_pool_nodes"], 60)
        self.assertEqual(metrics["val_nodes"], 20)
        self.assertEqual(metrics["test_nodes"], 20)


class InputViewTests(unittest.TestCase):
    def test_original_stage1_input_paths_use_selected_features_without_copying(self):
        spec = importlib.util.spec_from_file_location("movies_inputs", ROOT / "scripts/prepare_movies_inputs.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "MAGB/Movies"
            (source / "ImageFeature").mkdir(parents=True)
            (source / "TextFeature").mkdir()
            pd.DataFrame({"id": range(10), "label": [0] * 10}).to_csv(source / "Movies.csv", index=False)
            (source / "MoviesGraph.pt").write_bytes(b"graph")
            image = source / "ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy"
            text = source / "TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy"
            np.save(image, np.zeros((10, 4096), dtype=np.float32))
            np.save(text, np.ones((10, 4096), dtype=np.float32))
            before = file_sha256(text)
            work = root / "run"
            module.prepare(source.parent, work, text)
            expected = work / "data/Movies/TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy"
            self.assertTrue(expected.is_symlink())
            self.assertEqual(expected.resolve(), text.resolve())
            self.assertEqual(file_sha256(text), before)
            self.assertEqual((work / "stage1/config.json").resolve(), ROOT / "stage1/config.json")


if __name__ == "__main__":
    unittest.main()
