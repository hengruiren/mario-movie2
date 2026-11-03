import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stage1"))
sys.path.insert(0, str(ROOT))

from prototype_patch.class_guided import fit_clip_to_mario, map_clip_features, refine_consensus_batch
from prototype_patch.class_guided_assets import extract_all_clip, validate_clip_cache
from prototype_patch.class_guided_cli import parse_args
from prototype_patch.common import file_sha256, json_digest, save_tensor


class ClassGuidedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_agreement_and_both_acceptance_checks_with_hand_constructed_vectors(self):
        text = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [.8, .6], [1., 0.], [1., 0.]])
        images = torch.tensor([[2.4, 1.8], [1., 0.], [.8, .6], [.8, .6], [1., 0.], [.8, .6]])
        patches = torch.tensor([[[1., 0.], [.4, .916515]],
                                [[0., 1.], [0., 1.]],
                                [[.7, .714143], [.7, .714143]],
                                [[1., 0.], [1., 0.]],
                                [[1., 0.], [1., 0.]],
                                [[0., 0.], [0., 0.]]])
        out, d = refine_consensus_batch(text, images, patches, torch.eye(2))
        self.assertEqual(d["agreement"].tolist(), [True, False, True, True, True, True])
        self.assertEqual(d["accepted"].tolist(), [True, False, False, False, False, False])
        self.assertTrue(torch.equal(out[1:], images[1:]))
        self.assertTrue(torch.equal(d['patch_weights'][1], torch.zeros(2)))
        self.assertGreater(float(d['class_after'][3]), float(d['class_before'][3]))
        self.assertLess(float(d['text_after'][3]), float(d['text_before'][3]))
        torch.testing.assert_close(out[0].norm(), images[0].norm())
        expected = torch.softmax(torch.tensor([1., .4]) / .07, dim=0)
        torch.testing.assert_close(d['patch_weights'][0], expected, atol=1e-6, rtol=1e-5)

    def test_zero_weighted_sum_rejected_without_nan_or_mutation(self):
        t = torch.tensor([[1., 0.]])
        image = torch.tensor([[.8, .6]])
        out, d = refine_consensus_batch(t, image, torch.tensor([[[0., 1.], [0., -1.]]]), torch.eye(2))
        self.assertFalse(bool(d['candidate_valid'][0]))
        self.assertFalse(bool(d['accepted'][0]))
        self.assertTrue(torch.equal(out, image))

    def test_own_text_changes_patch_weights_but_preserves_class_acceptance_and_disagreement(self):
        text = torch.tensor([[.8, .6], [-.8, .6]])
        image = torch.tensor([[.6, .8], [.6, .8]])
        patches = torch.tensor([[[1., 0.], [.8, .6]], [[1., 0.], [.8, .6]]])
        prototypes = torch.tensor([[1., 0.], [-1., 0.]])
        _, by_class = refine_consensus_batch(text, image, patches, prototypes)
        out, by_text = refine_consensus_batch(text, image, patches, prototypes, patch_query='text')
        self.assertGreater(float(by_class['patch_weights'][0, 0]), float(by_class['patch_weights'][0, 1]))
        self.assertGreater(float(by_text['patch_weights'][0, 1]), float(by_text['patch_weights'][0, 0]))
        torch.testing.assert_close(by_text['patch_weights'][0], torch.softmax(torch.tensor([.8, 1.]) / .07, 0))
        self.assertEqual(by_text['agreement'].tolist(), [True, False])
        self.assertEqual(by_class['accepted'].tolist(), [False, False])
        self.assertEqual(by_text['accepted'].tolist(), [True, False])
        self.assertGreater(float(by_text['class_after'][0]), float(by_text['class_before'][0]) + 1e-6)
        self.assertTrue(torch.equal(out[1], image[1]))
        self.assertTrue(torch.equal(by_text['patch_weights'][1], torch.zeros(2)))

    def test_bridge_uses_only_training_full_image_pairs(self):
        rng = torch.Generator().manual_seed(12)
        x = F.normalize(torch.randn(30, 6, generator=rng), dim=-1)
        rotation, _ = torch.linalg.qr(torch.randn(6, 6, generator=rng))
        y = x @ rotation
        a = fit_clip_to_mario(x, y, torch.arange(20), ridge=.001)
        changed_x, changed_y = x.clone(), y.clone()
        changed_x[20:] = torch.randn(10, 6, generator=rng) * 100
        changed_y[20:] = torch.randn(10, 6, generator=rng) * 100
        b = fit_clip_to_mario(changed_x, changed_y, torch.arange(20), ridge=.001)
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]))
        mapped = map_clip_features(x[20:], a)
        self.assertGreater(float((mapped * y[20:]).sum(-1).mean()), .999)

    def test_cli_rejects_invalid_threshold_and_export_missing_inputs(self):
        common = ['--data-root', '/tmp/data', '--movies-subset', '/tmp/manifest', '--output-dir', '/tmp/output']
        for more in [['--mode', 'prepare', '--patch-temperature', '0'],
                     ['--mode', 'prepare', '--bridge-ridge', 'nan'],
                     ['--mode', 'prepare', '--min-class-gain', '-1'], []]:
            with self.assertRaises(SystemExit):
                parse_args(common + more)

    def test_real_small_clip_extraction_covers_all_nodes_and_detects_changed_image(self):
        from PIL import Image
        from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel, CLIPTextConfig, CLIPVisionConfig
        config = CLIPConfig.from_text_vision_configs(
            CLIPTextConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                           num_hidden_layers=1, num_attention_heads=4),
            CLIPVisionConfig(image_size=32, patch_size=16, hidden_size=32,
                             intermediate_size=64, num_hidden_layers=1, num_attention_heads=4),
            projection_dim=512)
        model = CLIPModel(config)
        processor = CLIPImageProcessor(size={'shortest_edge': 32}, crop_size={'height': 32, 'width': 32})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'Movies.csv').write_text('csv')
            (root / 'MoviesGraph.pt').write_bytes(b'graph')
            for node in range(3):
                Image.new('RGB', (9, 7), (node * 40, 50, 120)).save(root / f'{node}.jpg')
            data = SimpleNamespace(directory=root, records=[{}, {}, {}], train_ids=torch.tensor([0]))
            cache = root / 'cache.pt'
            with mock.patch('transformers.CLIPProcessor.from_pretrained', return_value=processor), \
                    mock.patch('transformers.CLIPModel.from_pretrained', return_value=model):
                payload = extract_all_clip(data, root, 'tiny-local-test', cache, device='cpu', batch_size=3)
            self.assertEqual(payload['node_ids'].tolist(), [0, 1, 2])
            self.assertEqual(payload['patches'].shape, (3, 4, 512))
            self.assertEqual(payload['global_features'].shape, (3, 512))
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            with mock.patch('transformers.CLIPModel.from_pretrained') as loader:
                reused = extract_all_clip(data, root, 'tiny-local-test', cache, device='cpu')
                loader.assert_not_called()
            torch.testing.assert_close(reused['patches'], payload['patches'])
            Image.new('RGB', (9, 7), 'red').save(root / '1.jpg')
            with self.assertRaisesRegex(ValueError, 'cache settings differ'):
                extract_all_clip(data, root, 'tiny-local-test', cache, device='cpu')
            payload['node_ids'] = torch.tensor([0, 2, 1])
            with self.assertRaisesRegex(ValueError, 'ALL nodes'):
                validate_clip_cache(payload, data)

    def test_real_export_cli_and_stage2_loader_keep_text_and_reject_bad_provenance(self):
        import dgl
        from stage2.mario_llm_stage2.subset import make_manifest, save_manifest
        from stage2.mario_llm_stage2.data import load_movies
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'MAGB/Movies'
            directory.mkdir(parents=True)
            descriptions = json.loads((ROOT / 'configs/movies_classes.json').read_text())
            names = [r['name'] for r in descriptions]
            labels = torch.arange(100) % 20
            with (directory / 'Movies.csv').open('w') as handle:
                w = csv.DictWriter(handle, fieldnames=['id', 'label', 'second_category', 'text'])
                w.writeheader()
                for node, label in enumerate(labels.tolist()):
                    w.writerow(dict(id=node, label=label, second_category=names[label], text='sample'))
            graph = dgl.graph((torch.arange(99), torch.arange(1, 100)), num_nodes=100)
            graph.ndata['label'] = labels
            dgl.save_graphs(str(directory / 'MoviesGraph.pt'), [graph])
            manifest = root / 'targets.json'
            save_manifest(manifest, make_manifest(labels.numpy(), directory))
            rng = torch.Generator().manual_seed(42)
            prototypes = torch.eye(512)[:20]
            images = F.normalize(prototypes[labels] + .015 * torch.randn(100, 512, generator=rng), dim=-1)
            source = torch.stack([prototypes[labels], images], dim=1)
            source_path = root / 'source.pth'
            save_tensor(source_path, source)
            class_path = root / 'prototypes.pt'
            save_tensor(class_path, dict(prototypes=prototypes, class_names=names,
                        text_encoder_sha256='frozen-text-test', class_cache_sha256='class-cache-test'))
            signature = dict(csv_sha256=file_sha256(directory / 'Movies.csv'),
                             graph_sha256=file_sha256(directory / 'MoviesGraph.pt'),
                             stage2_manifest_sha256=file_sha256(manifest), class_cache_sha256='class-cache-test')
            Path(str(source_path) + '.json').write_text(json.dumps(dict(
                feature_sha256=file_sha256(source_path), frozen_text_sha256='frozen-text-test', signature=signature)))
            cache = root / 'cache.pt'
            save_tensor(cache, dict(global_features=images, patches=prototypes[labels, None].repeat(1, 4, 1),
                        node_ids=torch.arange(100), metadata=dict(kind='clip_all_nodes_global_and_2x2', grid=[2, 2],
                        csv_sha256=signature['csv_sha256'], graph_sha256=signature['graph_sha256'],
                        node_ids_sha256=json_digest(list(range(100))))))
            out = root / 'refined'
            command = [sys.executable, str(ROOT / 'scripts/class_guided_patch.py'), '--mode', 'export',
                       '--data-root', str(directory.parent), '--movies-subset', str(manifest),
                       '--stage1-feature', str(source_path), '--class-prototypes', str(class_path),
                       '--clip-cache', str(cache), '--output-dir', str(out), '--device', 'cpu']
            import os
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', DGLBACKEND='pytorch')
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr[-2000:])
            final_path = out / 'Movies_stage1_mlp.pth'
            final = torch.load(final_path, weights_only=True)
            self.assertTrue(torch.equal(source[:, 0], final[:, 0]))
            self.assertTrue(torch.isfinite(final).all())
            decisions = [json.loads(x) for x in (out / 'node_decisions.jsonl').read_text().splitlines()]
            self.assertEqual(len(decisions), 100)
            self.assertGreater(sum(d['accepted'] for d in decisions), 0)
            for d in decisions:
                self.assertNotIn('label', d)
                if not d['accepted']:
                    self.assertTrue(torch.equal(final[d['node'], 1], source[d['node'], 1]))
                else:
                    self.assertGreater(d['class_after'], d['class_before'] + 1e-6)
                    self.assertGreaterEqual(d['text_after'], d['text_before'])
            args = SimpleNamespace(data_root=str(directory.parent), dataset='Movies', seed=42,
                                   movies_subset=str(manifest), stage1_feature=str(final_path))
            stage2 = load_movies(args)
            self.assertEqual(tuple(stage2.features.shape), (100, 2, 512))
            config = json.loads((out / 'run_config.json').read_text())
            self.assertEqual(config['bridge_training_nodes'], 60)
            self.assertFalse(config['decision_uses_labels'])
            text_out = root / 'text-refined'
            text_command = command.copy()
            text_command[text_command.index('--output-dir') + 1] = str(text_out)
            text_command += ['--patch-query', 'text', '--bridge-checkpoint', str(out / 'clip_to_mario_bridge.pt')]
            text_result = subprocess.run(text_command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(text_result.returncode, 0, text_result.stdout[-2000:] + text_result.stderr[-2000:])
            text_config = json.loads((text_out / 'run_config.json').read_text())
            self.assertEqual(text_config['patch_weighting_query'], 'own_text_embedding')
            original_bridge = torch.load(out / 'clip_to_mario_bridge.pt', weights_only=True)
            text_bridge = torch.load(text_out / 'clip_to_mario_bridge.pt', weights_only=True)
            for key in ['weight', 'input_mean', 'target_mean']:
                self.assertTrue(torch.equal(original_bridge[key], text_bridge[key]))
            repeated = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn('fresh refinement output', repeated.stderr)
            Path(str(source_path) + '.json').write_text('{}')
            command[command.index('--output-dir') + 1] = str(root / 'bad')
            bad = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn('feature hash differs', bad.stderr)


if __name__ == '__main__':
    unittest.main()
