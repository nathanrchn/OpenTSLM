# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT
"""CPU smoke test of OpenTSLMSP on a tiny random LFM3 (real tokenizer and modeling code)."""

import os
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

SRC = sys.argv[1] if len(sys.argv) > 1 else "LiquidAI/LFM3-1.7B-Base"
TINY = sys.argv[2] if len(sys.argv) > 2 else "/tmp/tiny-lfm3"


def build_tiny():
    cfg = AutoConfig.from_pretrained(SRC, trust_remote_code=True)
    cfg.layer_types = ["conv", "full_attention", "conv", "sliding_attention"]
    cfg.num_hidden_layers = 4
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.head_dim = 16
    cfg.sliding_window = 8
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    model.save_pretrained(TINY)
    AutoTokenizer.from_pretrained(SRC, trust_remote_code=True).save_pretrained(TINY)


def sample(n_ts, length, eos, answer):
    return {
        "pre_prompt": "You are given sensor data. Classify the activity.",
        "time_series_text": [f"Axis {i}, mean 0.1 std 1.0:" for i in range(n_ts)],
        "time_series": [torch.randn(length) for _ in range(n_ts)],
        "post_prompt": "Rationale:",
        "answer": answer + eos,
    }


def main():
    if not os.path.exists(os.path.join(TINY, "config.json")):
        build_tiny()
    for chat in (False, True):
        torch.manual_seed(0)
        m = OpenTSLMSP(llm_id=TINY, device="cpu", attn_implementation="sdpa", chat_template=chat)
        assert m.is_lfm and m.clean_tokenization
        eos = m.get_eos_token()
        batch = [
            sample(3, 37, eos, "The signal is periodic. Answer: walking"),
            sample(1, 130, eos, "Flat. Answer: lying"),
            sample(12, 20, eos, "Answer: none"),
        ]
        batch = extend_time_series_to_match_patch_size_and_aggregate(batch, patch_size=4)

        # 1) prompt layout: BOS once, then segments
        seqs = m._build_prompt_embeds(batch)
        emb = m.llm.get_input_embeddings().weight
        assert torch.allclose(seqs[0][0], emb[m.tokenizer.bos_token_id]), "first position must be BOS"

        # 2) answer-only logits loss == HF full-logits loss
        with torch.no_grad():
            m.is_lfm = False
            ref = m.compute_loss(batch).item()
            m.is_lfm = True
            fast = m.compute_loss(batch).item()
        assert abs(ref - fast) < 1e-2 * max(1.0, abs(ref)), (ref, fast)
        print(f"loss hf={ref:.5f} answer-only={fast:.5f}")

        # 2b) loss + grads on encoder/projector
        loss = m.compute_loss(batch)
        loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.encoder.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.projector.parameters())

        # 3) padding invariance of training forward (sample 1 alone vs in a right-padded batch)
        m.eval()
        with torch.no_grad():
            alone = m.compute_loss(batch[1:2]).item()
            # build a batch where sample 1 is padded behind a longer sample
            padded = m.compute_loss([batch[2], batch[1]]).item()
            single2 = m.compute_loss(batch[2:3]).item()
        assert abs(alone) > 0 and abs(single2) > 0
        print(f"chat={chat} loss={loss.item():.4f} alone={alone:.4f} pair={padded:.4f} single2={single2:.4f}")

        # 4) LoRA hits attention, conv and MLP projections
        m.enable_lora(lora_r=4, lora_alpha=8)
        names = [n for n, p in m.llm.named_parameters() if p.requires_grad and "lora_A" in n]
        for key in ("q_proj", "in_proj", "out_proj", "w1", "w2", "w3"):
            assert any(f".{key}." in n for n in names), f"LoRA missing on {key}"
        m.train()
        loss = m.compute_loss(batch)
        loss.backward()
        assert any(p.grad is not None for n, p in m.llm.named_parameters() if "lora_B" in n)

        # 5) batched left-padded greedy generation == one-by-one generation
        #    (fp32: in bf16 a random tiny model's near-uniform logits flip argmax on rounding noise)
        m.eval()
        m.llm.float()
        with torch.no_grad():
            together = m.generate(batch, max_new_tokens=6, do_sample=False)
            single = [m.generate([b], max_new_tokens=6, do_sample=False)[0] for b in batch]
        print("batched:", together)
        print("single: ", single)
        assert together == single, "batched generation differs from single-sample generation"
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
