import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]


class OriginalRuntimeTests(unittest.TestCase):
    def test_official_stage1_and_model_source_audit(self):
        spec = importlib.util.spec_from_file_location('original_audit', ROOT / 'scripts/audit_original.py')
        audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
        result = audit.audit()
        self.assertEqual(result['original_stage1_files'], 22)
        self.assertTrue(result['original_model_recovered_after_reversing_compatibility'])

    def test_real_generation_with_embedding_prefix_and_updated_positions(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.set_num_threads(1)
        config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                             pad_token_id=0, eos_token_id=None, bos_token_id=1)
        llm = LlamaForCausalLM(config).eval()
        tree = ast.parse((ROOT / 'stage2/mario_llm_stage2/model.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MarioLLM')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_generate_with_template')
        namespace = {'torch': torch}
        exec(compile(ast.Module(body=[method], type_ignores=[]), 'model.py', 'exec'), namespace)
        inputs = torch.randn(7, 32)
        tokenizer = types.SimpleNamespace(pad_token_id=0, eos_token_id=None,
                                          decode=lambda ids, **kwargs: str(ids.tolist()))
        model = types.SimpleNamespace(llm=llm, device=torch.device('cpu'), tokenizer=tokenizer,
                                      max_new_tokens=4, build_template=lambda *args: (inputs, None, [], []))
        self.assertEqual(len(ast.literal_eval(namespace['_generate_with_template'](model, 0, 'mm'))), 4)

    def test_real_official_stage1_trains_full_split_then_hands_off_to_twenty_percent(self):
        import dgl
        sys.path.insert(0, str(ROOT))
        from stage2.mario_llm_stage2.subset import make_manifest, save_manifest
        spec = importlib.util.spec_from_file_location('original_inputs', ROOT / 'scripts/prepare_movies_inputs.py')
        inputs = importlib.util.module_from_spec(spec); spec.loader.exec_module(inputs)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'MAGB/Movies'; directory.mkdir(parents=True)
            nodes = 100; labels = np.arange(nodes) % 4
            pd.DataFrame(dict(id=np.arange(nodes), label=labels, text=['item'] * nodes,
                second_category=['class' + str(i) for i in labels])).to_csv(directory / 'Movies.csv', index=False)
            g = dgl.graph((torch.arange(nodes), (torch.arange(nodes) + 1) % nodes), num_nodes=nodes)
            g.ndata['label'] = torch.tensor(labels)
            dgl.save_graphs(str(directory / 'MoviesGraph.pt'), [g])
            (directory / 'ImageFeature').mkdir(); (directory / 'TextFeature').mkdir()
            image = directory / 'ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy'
            text = directory / 'TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy'
            rng = np.random.RandomState(42)
            np.save(image, rng.randn(nodes, 4096).astype(np.float32))
            np.save(text, rng.randn(nodes, 4096).astype(np.float32))
            manifest = make_manifest(labels, directory)
            manifest_path = Path(tmp) / 'targets.json'; save_manifest(manifest_path, manifest)
            work = Path(tmp) / 'work'
            inputs.prepare(directory.parent, work, text)
            env = dict(os.environ, DATA_ROOT=str(work / 'data'), CUDA_VISIBLE_DEVICES='',
                       DGLBACKEND='pytorch', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            result = subprocess.run([sys.executable, str(ROOT / 'stage1/main.py'), '--dataset', 'Movies',
                    '--use_large_features', '--seed', '42', '--n_epochs', '1', '--batch_size', '20',
                    '--hidden_dim', '64', '--checkpoint_folder', str(work / 'check')],
                    cwd=work / 'stage1', env=env, text=True, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr[-2000:])
            train_only = torch.load(work / 'stage1/Movies_stage1_mlp_trainonly.pth', weights_only=True)
            train_nodes = (train_only.abs().sum((1, 2)) > 0).nonzero(as_tuple=True)[0].tolist()
            self.assertEqual(set(train_nodes), set(manifest['splits']['train']))
            self.assertEqual(len(train_nodes), 60)
            exported = work / 'stage1/Movies_stage1_mlp.pth'
            embeddings = torch.load(exported, weights_only=True)
            self.assertEqual(tuple(embeddings.shape), (100, 2, 64))
            self.assertTrue(bool((embeddings.norm(dim=-1) > 0).all()))
            # Import the real loader even if another test used dependency stubs.
            spec = importlib.util.spec_from_file_location('stage2.mario_llm_stage2.real_data',
                                                          ROOT / 'stage2/mario_llm_stage2/data.py')
            module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            args = types.SimpleNamespace(data_root=str(directory.parent), dataset='Movies', seed=42,
                      movies_subset=str(manifest_path), stage1_feature=str(exported))
            data = module.load_movies(args)
            self.assertEqual(len(data.train_idx), 60)
            self.assertEqual(len(data.train_target_idx), 12)
            self.assertEqual(len(data.val_idx), 20)
            self.assertEqual(len(data.test_idx), 20)


if __name__ == '__main__':
    unittest.main()
