from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stage2.mario_llm_stage2.compat import answer_logits_kwargs


class Stage2CompatibilityTests(unittest.TestCase):
    def test_argument_selection_supports_both_api_names_and_full_logits_fallback(self):
        def modern(inputs_embeds=None, logits_to_keep=0):
            pass

        def legacy(inputs_embeds=None, num_logits_to_keep=0):
            pass

        def full_only(inputs_embeds=None):
            pass

        self.assertEqual(answer_logits_kwargs(modern, 4), {"logits_to_keep": 4})
        self.assertEqual(answer_logits_kwargs(legacy, 4), {"num_logits_to_keep": 4})
        self.assertEqual(answer_logits_kwargs(full_only, 4), {})

    def test_real_llama_answer_logits_preserve_loss_and_input_gradients(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.set_num_threads(1)
        config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=1, num_attention_heads=4,
                             num_key_value_heads=2, max_position_embeddings=64)
        model = LlamaForCausalLM(config).eval()
        inputs = torch.randn(2, 8, 32, requires_grad=True)
        labels = torch.full((2, 8), -100, dtype=torch.long)
        labels[:, -3:] = torch.tensor([[5, 9, 2], [7, 11, 2]])
        full = model(inputs_embeds=inputs, use_cache=False).logits
        full_loss = F.cross_entropy(full[:, :-1].reshape(-1, 64), labels[:, 1:].reshape(-1))
        full_gradient = torch.autograd.grad(full_loss, inputs)[0]
        kept = model(inputs_embeds=inputs, use_cache=False,
                     **answer_logits_kwargs(LlamaForCausalLM.forward, 4)).logits
        self.assertEqual(kept.shape, (2, 4, 64))
        kept_loss = F.cross_entropy(kept[:, :-1].reshape(-1, 64), labels[:, -3:].reshape(-1))
        kept_gradient = torch.autograd.grad(kept_loss, inputs)[0]
        torch.testing.assert_close(kept_loss, full_loss)
        torch.testing.assert_close(kept_gradient, full_gradient)


if __name__ == "__main__":
    unittest.main()
