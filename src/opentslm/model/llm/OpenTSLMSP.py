# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence

try:
    from peft import get_peft_model, LoraConfig, TaskType

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft not available. LoRA fine-tuning will be disabled.")

from opentslm.model_config import ENCODER_OUTPUT_DIM
from .TimeSeriesLLM import TimeSeriesLLM
from ..encoder.TransformerCNNEncoder import TransformerCNNEncoder
from ..projector.MLPProjector import MLPProjector
from opentslm.prompt.full_prompt import FullPrompt
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)


class OpenTSLMSP(TimeSeriesLLM):
    def __init__(
        self,
        llm_id: str = "meta-llama/Llama-3.2-1B",
        device: str = "cuda",
        attn_implementation: str = "eager",
        chat_template: bool = False,
        max_patches: int = 1024,
        ts_encoder: str = "cnn",
        chronos_id: str = "amazon/chronos-2",
        encoder_trainable: bool = True,
    ):
        super().__init__(device)

        # 1) tokenizer (ensure pad_token exists). Segments are sliced assuming right padding.
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_id, use_fast=True, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # 2) load LLM
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_id,
            dtype=torch.bfloat16,
            device_map={"": device},
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        if len(self.tokenizer) > self.llm.get_input_embeddings().num_embeddings:
            self.llm.resize_token_embeddings(len(self.tokenizer))

        # LFM models: one BOS at the start of the sequence and no special tokens inside
        # text segments or answers (the original Llama/Gemma recipe adds BOS to every segment).
        model_type = getattr(self.llm.config, "model_type", "") or ""
        self.is_lfm = model_type.startswith("lfm")
        self.clean_tokenization = self.is_lfm
        # Wrap prompts in the ChatML user/assistant turn (non-thinking) for post-trained checkpoints.
        self.chat_template = chat_template
        self.last_num_tokens = 0
        # >1 upweights the label tokens after the final "Answer:" in the training target
        self.answer_weight = 1.0

        # 3) encoder + projector (now internal)
        self.ts_encoder = ts_encoder
        if ts_encoder == "chronos2":
            # Pretrained Chronos-2 encoder: series of one sample share a group, so leads/axes attend
            # to each other; one token per 16-point patch plus the [REG] summary token.
            from chronos import Chronos2Pipeline

            self.encoder = Chronos2Pipeline.from_pretrained(chronos_id, device_map=device).model
            self.encoder_context = self.encoder.chronos_config.context_length
            self.patch_size = self.encoder.chronos_config.input_patch_size
            enc_dim = self.encoder.config.d_model
        else:
            self.encoder = TransformerCNNEncoder(max_patches=max_patches).to(device)
            self.patch_size = 4
            enc_dim = ENCODER_OUTPUT_DIM
        self.encoder_trainable = encoder_trainable
        self.encoder.requires_grad_(encoder_trainable)
        self.projector = MLPProjector(
            enc_dim, self.llm.config.hidden_size, device=device
        ).to(device)

        # LoRA-related attributes
        self.lora_enabled = False
        self.original_llm = (
            None  # Keep reference to original model for backward compatibility
        )

        # Freeze the LLM backbone for SP model (internally)
        for p in self.llm.parameters():
            p.requires_grad = False

    def enable_lora(
        self,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
    ):
        """
        Enable LoRA fine-tuning for the LLM component.

        Args:
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            target_modules: List of module names to apply LoRA to. If None, uses defaults.
        """
        if not PEFT_AVAILABLE:
            raise RuntimeError(
                "peft package is required for LoRA fine-tuning. Please install with: pip install peft"
            )

        if self.lora_enabled:
            raise RuntimeError(
                "LoRA is already enabled. Call disable_lora() first if you want to reconfigure LoRA."
            )

        # Store reference to original model before applying LoRA
        self.original_llm = self.llm

        # Default target modules for common architectures
        if target_modules is None and self.is_lfm:
            # LFM: attention (q/k/v/o + output gate), short-conv in/out projections, SwiGLU MLP
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "in_proj",
                "out_proj",
                "w1",
                "w2",
                "w3",
            ]
        if target_modules is None:
            target_modules = [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

        # Create LoRA config
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        try:
            # Apply LoRA to the model
            self.llm = get_peft_model(self.llm, lora_config)
            self.lora_enabled = True

            # Print LoRA info
            lora_params = sum(
                p.numel()
                for name, p in self.llm.named_parameters()
                if p.requires_grad and "lora_" in name
            )
            trainable_params = sum(
                p.numel() for p in self.llm.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in self.llm.parameters())
            print(f"✅ LoRA enabled:")
            print(f"   LoRA parameters: {lora_params:,}")
            print(f"   Total trainable parameters: {trainable_params:,}")
            print(f"   Total parameters: {total_params:,}")
            print(f"   LoRA %: {100 * lora_params / total_params:.2f}%")
            print(f"   Trainable %: {100 * trainable_params / total_params:.2f}%")

        except Exception as e:
            print(f"❌ Failed to enable LoRA: {e}")
            print(
                "   This might be due to incompatible target modules for your model architecture."
            )
            print(
                "   Try specifying different target_modules or check your model's layer names."
            )
            raise

    def get_lora_parameters(self):
        """Get LoRA parameters for the optimizer."""
        if not self.lora_enabled:
            return []

        lora_params = []
        for name, param in self.llm.named_parameters():
            if param.requires_grad and "lora_" in name:
                lora_params.append(param)
        return lora_params

    def disable_lora(self):
        """Disable LoRA and revert to original frozen LLM."""
        if not self.lora_enabled:
            raise RuntimeError(
                "LoRA is not enabled. Cannot disable LoRA when it's not active."
            )

        if self.original_llm is not None:
            self.llm = self.original_llm
            self.original_llm = None

        self.lora_enabled = False
        print("✅ LoRA disabled, reverted to frozen LLM")

    def _tokenize(self, texts: List[str]):
        if self.clean_tokenization:
            return self.tokenizer(
                texts, return_tensors="pt", padding=True, add_special_tokens=False
            )
        return self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    def _build_prompt_embeds(self, batch: List[Dict[str, any]]) -> List[torch.Tensor]:
        """Returns one unpadded [L_i, H] prompt embedding per sample (see pad_and_apply_batch)."""
        device = self.device
        H = self.llm.config.hidden_size
        embed = self.llm.get_input_embeddings()

        # 1) Gather all texts
        all_texts: List[str] = []
        text_ptrs: List[Tuple[int, int]] = []
        ts_counts: List[int] = []
        for sample in batch:
            pre, post = sample["pre_prompt"], sample["post_prompt"]
            if self.chat_template:
                pre = "<|im_start|>user\n" + pre
                post = post + "<|im_end|>\n<|im_start|>assistant\n<think></think>"
            start = len(all_texts)
            all_texts.append(pre)
            all_texts.extend(sample["time_series_text"])
            all_texts.append(post)
            end = len(all_texts)
            text_ptrs.append((start, end))
            ts_counts.append(len(sample["time_series_text"]))

        # 2) Tokenize & embed all texts
        tok = self._tokenize(all_texts)
        input_ids = tok.input_ids.to(device, non_blocking=True)
        attn_mask = tok.attention_mask.to(device, non_blocking=True)
        text_embeds = embed(input_ids)  # [N_all, P_max, H]
        lengths = attn_mask.sum(dim=1).tolist()

        bos_embed = None
        if self.clean_tokenization and self.tokenizer.bos_token_id is not None:
            bos_embed = embed(
                torch.tensor([self.tokenizer.bos_token_id], device=device)
            )  # [1, H]

        # 3) Batch time-series encode & project
        ts_list: List[torch.Tensor] = []
        for sample in batch:
            for ts in sample["time_series"]:
                # ensure [T] → [T,1]
                if ts.dim() == 1:
                    ts = ts.unsqueeze(-1)
                ts_list.append(ts)

        if self.ts_encoder == "chronos2":
            ts_proj = self._encode_chronos(batch, text_embeds.dtype)
        else:
            # Encode series grouped by length so a sample's tokens never depend on the other
            # samples in the batch (padding to the batch max would add zero patches).
            ts_proj: List[torch.Tensor] = [None] * len(ts_list)
            by_len: Dict[int, List[int]] = {}
            for k, ts in enumerate(ts_list):
                by_len.setdefault(ts.size(0), []).append(k)
            for T, ks in by_len.items():
                x = torch.stack([ts_list[k] for k in ks]).to(device, non_blocking=True)
                rem = T % self.patch_size
                if rem:
                    x = torch.cat(
                        [x, x.new_zeros(x.size(0), self.patch_size - rem, x.size(2))], dim=1
                    )
                enc = self.encoder(x.squeeze(-1))  # [n, N_patches, embed_dim]
                proj = self.projector(enc).to(text_embeds.dtype)  # [n, N_patches, H]
                for j, k in enumerate(ks):
                    ts_proj[k] = proj[j]

        # 4) Re‐assemble per sample
        all_seq_embeds = []
        ts_offset = 0
        for (start, end), n_ts in zip(text_ptrs, ts_counts):
            seq_embeds = [] if bos_embed is None else [bos_embed]

            # pre_prompt
            seq_embeds.append(text_embeds[start, : lengths[start], :])

            # each (textᵢ, tsᵢ)
            for i in range(n_ts):
                idx = start + 1 + i
                seq_embeds.append(text_embeds[idx, : lengths[idx], :])
                seq_embeds.append(ts_proj[ts_offset + i])  # [N_patches, H]

            ts_offset += n_ts

            # post_prompt
            seq_embeds.append(text_embeds[end - 1, : lengths[end - 1], :])

            all_seq_embeds.append(torch.cat(seq_embeds, dim=0))

        return all_seq_embeds

    def _encode_chronos(self, batch, dtype) -> List[torch.Tensor]:
        """One projected token sequence per series: its patch embeddings plus the [REG] token."""
        series, groups = [], []
        for si, sample in enumerate(batch):
            for ts in sample["time_series"]:
                ts = torch.as_tensor(ts, dtype=torch.float32).reshape(-1)[-self.encoder_context :]
                series.append(ts)
                groups.append(si)
        t_max = max(len(x) for x in series)
        ctx = torch.full((len(series), t_max), float("nan"))
        for k, x in enumerate(series):  # left-pad with NaN (masked inside Chronos-2)
            ctx[k, t_max - len(x) :] = x
        with torch.set_grad_enabled(self.encoder_trainable and torch.is_grad_enabled()):
            hidden = self.encoder.encode(
                context=ctx.to(self.device), group_ids=torch.tensor(groups, device=self.device)
            )[0][0]  # [N, n_patches + REG + future, d]
        out = []
        for k, x in enumerate(series):
            n = -(-len(x) // self.patch_size)
            out.append(self.projector(hidden[k, -(n + 2) : -1].float()).to(dtype))
        return out

    def pad_and_apply_batch(
        self,
        batch: List[Dict[str, any]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TL;DR:
            This function is probably the most crucial part of OpenTSLM-SP, and also the hardest to understand.
            It's where the magic happens and legends are made.

            It batches and embeds all text and time series inputs in parallel,
            then reassembles them per sample to allow efficient GPU execution.
            Praise the PyTorch Wizards: ChatGPT-o4-mini-high, Patrick, and Thomas (listed in strictly descending order of skill).

        Long description:
            Processes a batch of training samples by embedding and aligning text and time series data
            for efficient parallel processing on the GPU.

            This method performs the following steps:

            1. Extracts all text components (pre_prompt, time_series_text, post_prompt) from each sample,
            and embeds them in a single batch using the LLM tokenizer and embedding layer. Padding and attention
            masks are applied to accommodate variable-length sequences.

            2. Gathers all time series segments across the batch and pads them
            into a single tensor of shape [N_ts_total, T_padded, D], where T_padded
            is the smallest multiple of `patch_size` ≥ the longest segment length.
            This tensor is then encoded and projected into the LLM hidden space.

            3. After all embeddings are extracted, the function reconstructs each original sample by interleaving its
            embedded pre_prompt, time series texts and corresponding time series embeddings, and the post_prompt, preserving original order.

            4. Pads all reassembled sequences to a uniform length across the batch to form the final input tensor
                and attention mask.

            5. All of this is only required for efficient processing.

        - pre_prompt: str
        - time_series_text: List[str]
        - time_series: Tensor [N_ts, T] or [N_ts, T, D]
        - post_prompt: str
        Returns (inputs_embeds, attention_mask)
        """
        all_seq_embeds = self._build_prompt_embeds(batch)
        all_seq_masks = [
            torch.ones(e.size(0), device=self.device, dtype=torch.long)
            for e in all_seq_embeds
        ]
        # 5) Batch-pad the final sequences (right padding)
        inputs_embeds = pad_sequence(all_seq_embeds, batch_first=True)  # [B, L_max, H]
        attention_mask = pad_sequence(all_seq_masks, batch_first=True)  # [B, L_max]

        return inputs_embeds, attention_mask

    def generate(
        self, batch: List[Dict[str, any]], max_new_tokens: int = 50, **generate_kwargs
    ) -> List[str]:
        # Left-pad so every row ends at its prompt and decoding continues from there.
        seqs = self._build_prompt_embeds(batch)
        B, L, H = len(seqs), max(e.size(0) for e in seqs), seqs[0].size(1)
        inputs_embeds = seqs[0].new_zeros(B, L, H)
        attention_mask = torch.zeros(B, L, device=self.device, dtype=torch.long)
        for i, e in enumerate(seqs):
            inputs_embeds[i, L - e.size(0) :] = e
            attention_mask[i, L - e.size(0) :] = 1

        generate_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        generate_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        gen_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )
        return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

    def compute_loss(self, batch: List[Dict[str, any]]) -> torch.Tensor:
        """
        batch: same format as generate()
        Each sample is laid out contiguously as [prompt, answer] and right-padded, so no padding
        sits between prompt and answer (matters for convolutional layers, e.g. LFM).
        """
        answers = [b["answer"] for b in batch]
        prompt_embeds = self._build_prompt_embeds(batch)

        ans_tok = self._tokenize(answers)
        ans_ids = ans_tok.input_ids.to(self.device, non_blocking=True)
        ans_len = ans_tok.attention_mask.sum(dim=1).tolist()
        ans_emb = self.llm.get_input_embeddings()(ans_ids)  # [B, A_max, H]

        seqs, labels, weights = [], [], []
        for i, p in enumerate(prompt_embeds):
            a = int(ans_len[i])
            w = torch.ones(p.size(0) + a, device=self.device)
            if self.answer_weight != 1.0 and "Answer:" in answers[i]:
                cut = answers[i].rindex("Answer:")
                n_pre = len(self._tokenize([answers[i][:cut]]).input_ids[0])
                w[p.size(0) + n_pre :] = self.answer_weight
            weights.append(w)
            seqs.append(torch.cat([p, ans_emb[i, :a]], dim=0))
            labels.append(
                torch.cat(
                    [
                        torch.full((p.size(0),), -100, device=self.device, dtype=torch.long),
                        ans_ids[i, :a],
                    ]
                )
            )
        masks = [torch.ones(x.size(0), device=self.device, dtype=torch.long) for x in seqs]
        inputs_embeds = pad_sequence(seqs, batch_first=True)
        attention_mask = pad_sequence(masks, batch_first=True)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        self.last_num_tokens = int(attention_mask.sum().item())

        if self.is_lfm:
            # Project only the supervised positions onto the vocabulary: the prompt (mostly
            # time-series tokens) never needs logits, which saves a lot of memory and compute.
            base = self.llm.get_base_model() if self.lora_enabled else self.llm
            hidden = base.model(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask
            ).last_hidden_state
            shift_labels = labels[:, 1:]
            keep = shift_labels != -100
            logits = base.lm_head(hidden[:, :-1][keep])
            ce = torch.nn.functional.cross_entropy(logits.float(), shift_labels[keep], reduction="none")
            w = pad_sequence(weights, batch_first=True)[:, 1:][keep]
            return (ce * w).sum() / w.sum()

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return outputs.loss

    def get_eos_token(self) -> str:
        return self.tokenizer.eos_token

    def store_to_file(self, path: str):
        checkpoint = {
            "encoder_state": self.encoder.state_dict(),
            "projector_state": self.projector.state_dict(),
        }

        # Add LoRA state to checkpoint
        self.save_lora_state_to_checkpoint(checkpoint)

        torch.save(checkpoint, path)

    def load_from_file(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(ckpt["encoder_state"])
        self.projector.load_state_dict(ckpt["projector_state"])

        # Load LoRA state if present (allow missing for backward compatibility)
        self.load_lora_state_from_checkpoint(ckpt, allow_missing=True)

        print(f"📥 Loaded model from epoch {ckpt.get('epoch', '?')}")

    def load_lora_state_from_checkpoint(
        self, checkpoint: dict, allow_missing: bool = False
    ):
        """
        Load LoRA adapters from a checkpoint.

        Args:
            checkpoint: Checkpoint dictionary containing potential LoRA state
            allow_missing: If True, don't raise exception when checkpoint has no LoRA but model expects it

        Raises:
            RuntimeError: When there's a mismatch between checkpoint and current LoRA state
        """
        checkpoint_has_lora = checkpoint.get("lora_enabled", False)

        if checkpoint_has_lora and "lora_state" in checkpoint:
            # Checkpoint has LoRA adapters
            if not self.lora_enabled:
                raise RuntimeError(
                    "Checkpoint contains LoRA adapters but LoRA is not currently enabled. "
                    "Call enable_lora() before loading this checkpoint."
                )

            # Load LoRA adapters
            try:
                lora_state = checkpoint["lora_state"]
                loaded_count = 0
                missing_keys = []

                # Track which LoRA parameters we expect to find
                expected_lora_params = {
                    name
                    for name, param in self.llm.named_parameters()
                    if param.requires_grad and "lora_" in name
                }

                for name, param in self.llm.named_parameters():
                    if name in lora_state and param.requires_grad and "lora_" in name:
                        param.data.copy_(lora_state[name])
                        loaded_count += 1
                    elif param.requires_grad and "lora_" in name:
                        missing_keys.append(name)

                if missing_keys and not allow_missing:
                    raise RuntimeError(
                        f"Could not find LoRA parameters in checkpoint: {missing_keys[:5]}..."
                    )

                print(f"📥 Loaded LoRA adapters: {loaded_count} parameters")
                return loaded_count

            except Exception as e:
                if "Could not find LoRA parameters" in str(e):
                    raise  # Re-raise our custom exception
                raise RuntimeError(f"Failed to load LoRA adapters: {e}")

        elif checkpoint_has_lora:
            raise RuntimeError(
                "Checkpoint indicates LoRA was enabled but no LoRA state found"
            )

        # Handle case where checkpoint has no LoRA but model expects it
        if not checkpoint_has_lora and self.lora_enabled:
            if not allow_missing:
                raise RuntimeError(
                    "Loading checkpoint from before LoRA was enabled, but LoRA is currently enabled. "
                    "LoRA adapters will be randomly initialized. Set allow_missing=True to allow this."
                )
            else:
                print("⚠️  Loading checkpoint from before LoRA was enabled.")
                print("   LoRA adapters will be randomly initialized.")

        return 0

    def save_lora_state_to_checkpoint(self, checkpoint: dict):
        """
        Save LoRA adapters to a checkpoint dictionary.

        Args:
            checkpoint: Checkpoint dictionary to add LoRA state to

        Returns:
            int: Number of LoRA parameters saved
        """
        checkpoint["lora_enabled"] = self.lora_enabled

        if self.lora_enabled and hasattr(self.llm, "peft_config"):
            try:
                # Save LoRA adapter weights
                lora_state = {}
                for name, param in self.llm.named_parameters():
                    if param.requires_grad and "lora_" in name:
                        lora_state[name] = param.data.clone()

                if lora_state:
                    checkpoint["lora_state"] = lora_state
                    checkpoint["lora_config"] = self.llm.peft_config
                    print(f"💾 Saved LoRA adapters with {len(lora_state)} parameters")
                    return len(lora_state)
            except Exception as e:
                raise RuntimeError(f"Failed to save LoRA adapters: {e}")

        return 0

    def eval_prompt(
        self, prompt: FullPrompt, max_new_tokens: int = 30000, normalize: bool = False
    ) -> str:
        """
        Evaluate a prompt and return the generated text.
        """

        batch = [prompt.to_dict()]
        self.eval()
        batch = extend_time_series_to_match_patch_size_and_aggregate(
            batch, normalize=normalize
        )
        output = self.generate(batch, max_new_tokens=max_new_tokens)
        return output[0]
