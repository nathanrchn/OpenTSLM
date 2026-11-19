# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT
"""
Test-set generation with vLLM for an exported OpenTSLM-SoftPrompt model (train_sp.py --export_dir).

Prompt embeddings are built with OpenTSLMSP._build_prompt_embeds itself (time-series encoder + projector +
the LLM's embedding table, no LLM weights loaded) and passed to vLLM as `prompt_embeds`; the LoRA-merged LLM
runs inside vLLM. Run one process per GPU with --shard/--num_shards, then --merge to join and score.
"""

import argparse
import contextlib
import json
import os
import pickle
import random
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_sp  # noqa: E402  (dataset classes, collate, budget, task table)


def build_embedder(export_dir, device):
    from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
    from opentslm.model.projector.MLPProjector import MLPProjector

    blob = torch.load(os.path.join(export_dir, "embedder.pt"), weights_only=False, map_location="cpu")
    cfg = blob["config"]
    llm_dir = os.path.join(export_dir, "llm")
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(llm_dir, trust_remote_code=True)
    except Exception:
        from transformers import PreTrainedTokenizerFast

        tc = json.load(open(os.path.join(llm_dir, "tokenizer_config.json")))
        tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(llm_dir, "tokenizer.json"),
                                      bos_token=tc.get("bos_token"), eos_token=tc.get("eos_token"),
                                      pad_token=tc.get("pad_token"))
    tok.padding_side = "right"

    class _EmbeddingOnlyLLM(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.emb = torch.nn.Embedding.from_pretrained(weight, freeze=True)
            self.config = SimpleNamespace(hidden_size=weight.shape[1])

        def get_input_embeddings(self):
            return self.emb

    emb = OpenTSLMSP.__new__(OpenTSLMSP)
    torch.nn.Module.__init__(emb)
    emb.device = device
    emb.tokenizer = tok
    emb.llm = _EmbeddingOnlyLLM(blob["embed_tokens"]).to(device)
    emb.clean_tokenization, emb.chat_template, emb.lora_enabled = True, False, False
    emb.ts_encoder = cfg["ts_encoder"]
    emb.encoder_trainable = False
    if cfg["ts_encoder"] != "chronos2":
        raise SystemExit("vllm_eval supports the chronos2 encoder only")
    from chronos import Chronos2Pipeline

    emb.encoder = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device).model
    if blob.get("encoder_state"):
        emb.encoder.load_state_dict(blob["encoder_state"])
    emb.encoder.eval()
    emb.encoder_context = emb.encoder.chronos_config.context_length
    emb.patch_size = emb.encoder.chronos_config.input_patch_size
    emb.merge_tokens = cfg["merge_tokens"]
    emb.projector = MLPProjector(emb.encoder.config.d_model * emb.merge_tokens, cfg["hidden_size"], device=device)
    emb.projector.load_state_dict(blob["projector_state"])
    emb.projector.eval()
    return emb, tok, cfg


def record(sample, pred, idx):
    rec = {"idx": idx, "pre_prompt": sample["pre_prompt"], "time_series_text": sample["time_series_text"],
           "post_prompt": sample["post_prompt"], "generated": pred, "gold": sample["answer"]}
    for k in ("template_id", "ecg_id", "label"):
        if k in sample and sample[k] is not None:
            rec[k] = sample[k]
    return rec


def generate(args):
    from vllm import LLM, SamplingParams

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb, tok, cfg = build_embedder(args.export_dir, device)
    train_sp.OPTS.update({"collate_patch": 1, "ecg_hz": cfg["ecg_hz"], "sleep_uv": cfg["sleep_uv"]})
    cache = os.path.join(args.test_cache, f"{args.stage}.pkl") if args.test_cache else None
    if cache and os.path.exists(cache):  # written by train_sp.py --dump_test: no dataset rebuild
        t0 = time.time()
        with open(cache, "rb") as f:
            blob = pickle.load(f)
        test_ds, budget = blob["samples"], blob["budget"]
        print(json.dumps({"event": "test_cache_loaded", "n": len(test_ds), "sec": round(time.time() - t0, 1)}), flush=True)
    else:
        cls = train_sp.dataset_class(args.stage)
        quiet = args.stage == "stage5_ecg_cot"
        with (contextlib.redirect_stdout(open(os.devnull, "w")) if quiet else contextlib.nullcontext()):
            train_ds, test_ds = cls("train", EOS_TOKEN=tok.eos_token), cls("test", EOS_TOKEN=tok.eos_token)
        budget = train_sp.answer_token_budget(SimpleNamespace(tokenizer=tok), train_ds)[0]

    # same sample selection and ordering as train_sp.run_test
    idxs = list(range(len(test_ds)))
    if args.test_samples:
        idxs = random.Random(0).sample(idxs, min(args.test_samples, len(idxs)))

    def size(i):
        s = test_ds[i]
        return sum(len(t) for t in s["time_series"]) + len(s["pre_prompt"]) + len(s["post_prompt"])

    idxs.sort(key=size, reverse=True)
    idxs = idxs[args.shard :: args.num_shards]

    llm = LLM(model=os.path.join(args.export_dir, "llm"), dtype="bfloat16", enable_prompt_embeds=True,
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len, trust_remote_code=True,
              seed=0)
    eos = tok.eos_token_id
    main_sp = SamplingParams(temperature=0.0, max_tokens=budget, stop_token_ids=[eos], skip_special_tokens=True)
    fix_sp = SamplingParams(temperature=0.0, max_tokens=48, stop_token_ids=[eos], skip_special_tokens=True)
    force = args.force_answer and train_sp.STAGE_DEFAULTS[args.stage]["task"] in ("har", "sleep", "ecg")

    @torch.no_grad()
    def embeds_for(samples):  # small encoder batches: vLLM already holds most of the GPU
        out = []
        for b in range(0, len(samples), args.embed_bs):
            out += [{"prompt_embeds": e.to("cpu")} for e in emb._build_prompt_embeds(samples[b : b + args.embed_bs])]
        return out

    out_path = f"{args.out}.shard{args.shard}"
    t0, done = time.time(), 0
    with open(out_path, "w", encoding="utf-8") as f:
        for b in range(0, len(idxs), args.chunk):
            chunk = idxs[b : b + args.chunk]
            batch = train_sp.collate([test_ds[i] for i in chunk])
            outs = llm.generate(embeds_for(batch), main_sp, use_tqdm=False)
            preds = [o.outputs[0].text for o in outs]
            if force:
                missing = [j for j, p in enumerate(preds) if "Answer:" not in p]
                if missing:
                    cont = []
                    for j in missing:
                        smp = dict(batch[j])
                        smp["post_prompt"] = smp["post_prompt"] + preds[j].rstrip() + " Answer:"
                        cont.append(smp)
                    conts = llm.generate(embeds_for(cont), fix_sp, use_tqdm=False)
                    for j, c in zip(missing, conts):
                        preds[j] = preds[j].rstrip() + " Answer:" + c.outputs[0].text.split("\n")[0]
            for i, sample, pred in zip(chunk, batch, preds):
                f.write(json.dumps(record(sample, pred, i), ensure_ascii=False) + "\n")
            done += len(chunk)
            print(json.dumps({"event": "vllm_progress", "shard": args.shard, "done": done, "total": len(idxs),
                              "sec": round(time.time() - t0, 1)}), flush=True)
    print(json.dumps({"event": "vllm_shard_done", "shard": args.shard, "n": done, "sec": round(time.time() - t0, 1),
                      "max_new_tokens": budget}), flush=True)


def merge(args):
    rows = []
    for s in range(args.num_shards):
        with open(f"{args.out}.shard{s}", encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    rows.sort(key=lambda r: r["idx"])
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    task = train_sp.STAGE_DEFAULTS[args.stage]["task"]
    if task == "ecg":  # the ECG scorer needs the template answer table loaded
        train_sp.dataset_class(args.stage).get_possible_answers_for_template(1)
    metrics = {"n": len(rows), **train_sp.score.score(task, args.out)}
    if args.compare:  # agreement with HF generation on the same samples (matched by idx)
        hf = {}
        with open(args.compare, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                hf[r["idx"]] = r["generated"]
        common = [r for r in rows if r["idx"] in hf]
        ans = lambda t: t.split("Answer:")[-1].strip()[:60]
        metrics["hf_compare"] = {"n": len(common),
                                 "exact_text": sum(r["generated"] == hf[r["idx"]] for r in common) / max(1, len(common)),
                                 "same_answer": sum(ans(r["generated"]) == ans(hf[r["idx"]]) for r in common) / max(1, len(common))}
    json.dump(metrics, open(args.out.replace(".jsonl", "") + "_metrics.json", "w"), indent=2)
    print(json.dumps({"event": "vllm_metrics", **metrics}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export_dir")
    ap.add_argument("--stage", default="stage5_ecg_cot")
    ap.add_argument("--out", required=True, help="merged predictions path (shards are <out>.shardN)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--test_samples", type=int, default=None)
    ap.add_argument("--force_answer", action="store_true")
    ap.add_argument("--chunk", type=int, default=256, help="samples per vLLM generate call")
    ap.add_argument("--gpu_mem", type=float, default=0.75)
    ap.add_argument("--embed_bs", type=int, default=16, help="samples per time-series encoder call")
    ap.add_argument("--test_cache", default=None, help="dir with <stage>.pkl from train_sp.py --dump_test")
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--merge", action="store_true", help="join shards and score")
    ap.add_argument("--compare", default=None, help="HF test_predictions.jsonl to compare against")
    a = ap.parse_args()
    merge(a) if a.merge else generate(a)
