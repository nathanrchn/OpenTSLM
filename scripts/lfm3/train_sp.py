# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT
"""
Budgeted OpenTSLM-SoftPrompt curriculum trainer (used for the LFM3-1.7B runs).

Differences to curriculum_learning.py, all aimed at bounding GPU cost:
- per-stage epoch caps, validation on a fixed subset, early stopping with small patience
- resumable: per-stage best/last checkpoints, finished stages are skipped
- batched, left-padded greedy generation for the test sets with a per-stage token cap
- optional stratified subsampling of the ECG-QA-CoT training split
- scores test predictions with the paper's own parsers (scripts/lfm3/score.py)
"""

import argparse
import json
import math
import re
import os
import pickle
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from transformers import get_linear_schedule_with_warmup

from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.model_config import GRAD_CLIP_NORM, PATCH_SIZE, WARMUP_FRAC, WEIGHT_DECAY
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score  # noqa: E402

LR_ENCODER = 2e-4
LR_PROJECTOR = 1e-4  # curriculum_learning.py also uses this LR for LoRA
STAGE_ORDER = [
    "stage1_mcq",
    "stage2_captioning",
    "stage3_cot",
    "stage4_sleep_cot",
    "stage5_ecg_cot",
]
STAGE_DEFAULTS = {
    "stage1_mcq": dict(epochs=3, lora=False, task="tsqa"),
    "stage2_captioning": dict(epochs=2, lora=False, task=None),
    "stage3_cot": dict(epochs=3, lora=True, task="har"),
    "stage4_sleep_cot": dict(epochs=20, lora=True, task="sleep"),
    "stage5_ecg_cot": dict(epochs=2, lora=True, task="ecg"),
}


OPTS = {}  # dataset options set from the command line (see main)
FORCE_ANSWER = False  # if a generation ends without "Answer:", append it and let the model finish
RANK, WORLD = 0, 1  # data-parallel rank / world size (torchrun sets WORLD_SIZE)


def barrier():
    if WORLD > 1:
        dist.barrier()


def bcast(x):
    """Broadcast a python float from rank 0."""
    if WORLD == 1:
        return x
    t = torch.tensor([float("nan") if x is None else float(x)], device=f"cuda:{torch.cuda.current_device()}"
                     if torch.cuda.is_available() else "cpu")
    dist.broadcast(t, 0)
    return t.item()


def sync_grads(params):
    """Average gradients across data-parallel ranks (one flat all-reduce)."""
    if WORLD == 1:
        return
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    flat = torch.cat([g.reshape(-1).float() for g in grads])
    dist.all_reduce(flat)
    flat /= WORLD
    off = 0
    for p, g in zip(params, grads):
        n = g.numel()
        p.grad = flat[off : off + n].view_as(p).to(p.dtype)
        off += n


def log(out_dir, **kw):
    if RANK != 0:
        return
    kw["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    line = json.dumps(kw, default=str)
    print(line, flush=True)
    with open(os.path.join(out_dir, "log.jsonl"), "a") as f:
        f.write(line + "\n")


def dataset_class(stage):
    if stage == "stage1_mcq":
        from opentslm.time_series_datasets.TSQADataset import TSQADataset

        return TSQADataset
    if stage == "stage2_captioning":
        from opentslm.time_series_datasets.m4.M4QADataset import M4QADataset

        return M4QADataset
    if stage == "stage3_cot":
        from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset

        return HARCoTQADataset
    if stage == "stage4_sleep_cot":
        from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import (
            SleepEDFCoTQADataset,
        )

        if not OPTS.get("sleep_uv"):
            return SleepEDFCoTQADataset
        from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt

        class SleepUV(SleepEDFCoTQADataset):
            """States the EEG amplitude in microvolts. The stock prompt prints volts with 4 decimals
            ("std 0.0001"), so the model never sees the amplitude that separates deep sleep from REM/N1."""

            def _get_text_time_series_prompt_list(self, row):
                series = np.array(row["time_series"], dtype=np.float32)
                mean = float(np.mean(series))
                std = max(float(np.std(series)), 1e-6)
                norm = (series - mean) / std
                text = (f"The following is the EEG time series, it has mean {mean * 1e6:.1f} microvolts "
                        f"and std {std * 1e6:.1f} microvolts:")
                return [TextTimeSeriesPrompt(text, norm.tolist())]

        return SleepUV
    if stage == "stage5_ecg_cot":
        from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
        from opentslm.time_series_datasets.ecg_qa.ECGQACoTQADataset import (
            ECGQACoTQADataset,
        )

        class ECGWithIds(ECGQACoTQADataset):
            """Keeps template_id / ecg_id on formatted samples (needed by the paper's ECG scorer).
            With OPTS["ecg_hz"] == 500 the PTB-XL high-rate signal is used without the 5x downsampling,
            and each record's 12 leads are stored once (float16) and shared by all its questions."""

            _shared = {}
            _prompt_cache = {}

            def _get_text_time_series_prompt_list(self, row):
                # 500 Hz: build each record's 12 lead prompts once (float16), not once per question;
                # the per-question list->array round trips made loading take ~14 min.
                if OPTS.get("ecg_hz", 100) != 500:
                    return super()._get_text_time_series_prompt_list(row)
                key = str(row.get("ecg_paths") or row.get("ecg_id"))
                hit = ECGWithIds._prompt_cache.get(key)
                if hit is None:
                    hit = [TextTimeSeriesPrompt(p.get_text(), p.get_time_series().astype(np.float16))
                           for p in super()._get_text_time_series_prompt_list(row)]
                    ECGWithIds._prompt_cache[key] = hit
                return hit

            @classmethod
            def _process_ecg_lead(cls, ecg_path, lead_idx):
                if OPTS.get("ecg_hz", 100) != 500:
                    return super()._process_ecg_lead(ecg_path, lead_idx)
                key = f"{ecg_path}:lead_{lead_idx}"
                if key not in cls._processed_ecg_cache:
                    sig, _, _ = cls._load_ecg_data(ecg_path)
                    lead = sig[:, lead_idx] if sig.ndim > 1 else sig
                    mean, std = float(np.mean(lead)), float(np.std(lead))
                    norm = (lead - mean) / std if std > 1e-6 else lead - mean
                    cls._processed_ecg_cache[key] = (norm.astype(np.float32), mean, std)
                return cls._processed_ecg_cache[key]

            def _format_sample(self, row):
                sample = super()._format_sample(row)
                if OPTS.get("ecg_hz", 100) == 500:
                    key = str(row.get("ecg_paths") or row.get("ecg_id"))
                    arr = ECGWithIds._shared.get(key)
                    if arr is None:
                        arr = np.stack([np.asarray(t, dtype=np.float16) for t in sample["time_series"]])
                        ECGWithIds._shared[key] = arr
                    sample["time_series"] = arr
                else:
                    # float32 halves host RAM (232k samples x 12 leads x 1000 points)
                    sample["time_series"] = [np.asarray(t, dtype=np.float32) for t in sample["time_series"]]
                sample["template_id"] = row.get("template_id")
                ecg_id = row.get("ecg_id")
                if isinstance(ecg_id, (list, tuple)):
                    ecg_id = ecg_id[0] if ecg_id else None
                sample["ecg_id"] = ecg_id
                return sample

        return ECGWithIds
    raise ValueError(stage)


def ensure_ecg_downloaded():
    from opentslm.time_series_datasets.ecg_qa.ecgqa_cot_loader import download_ecg_qa_cot_if_not_exists
    from opentslm.time_series_datasets.ecg_qa.ecgqa_loader import (
        download_ecg_qa_if_not_exists,
        download_ptbxl_if_not_exists,
    )

    download_ecg_qa_if_not_exists()
    download_ptbxl_if_not_exists()
    download_ecg_qa_cot_if_not_exists()


def final_label(answer: str) -> str:
    label = re.sub(r"<\|.*?\|>", "", answer.split("Answer:")[-1]).strip().rstrip(".").lower()
    return label[:80]


def stratified_indices(dataset, fraction, seed, part=0):
    """Stratified (template, answer) subsample. part=k returns the k-th disjoint slice of the same
    shuffle, so part=1 is fresh data relative to part=0. Grouping key kept as in the first runs."""
    groups = defaultdict(list)
    for i in range(len(dataset)):
        s = dataset[i]
        groups[(s.get("template_id"), s["answer"].split("Answer:")[-1].strip().lower()[:80])].append(i)
    rng = random.Random(seed)
    keep = []
    for key in sorted(groups, key=str):
        idxs = groups[key]
        rng.shuffle(idxs)
        m = max(1, round(len(idxs) * fraction))
        keep.extend(idxs[part * m : (part + 1) * m])
    rng.shuffle(keep)
    return keep


def collate(batch):
    # Chronos-2 handles any length itself (OPTS["collate_patch"]=1: no zero padding).
    # Shallow copies: collate replaces "time_series" and must not touch the dataset's items.
    return extend_time_series_to_match_patch_size_and_aggregate([dict(b) for b in batch],
                                                                patch_size=OPTS.get("collate_patch", PATCH_SIZE))


def trainable_state(model):
    state = {"projector_state": {k: v.detach().cpu().clone() for k, v in model.projector.state_dict().items()}}
    if model.encoder_trainable:  # a frozen pretrained encoder is not saved
        state["encoder_state"] = {k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()}
    state["lora_enabled"] = model.lora_enabled
    if model.lora_enabled:
        state["lora_state"] = {
            n: p.detach().cpu().clone()
            for n, p in model.llm.named_parameters()
            if p.requires_grad and "lora_" in n
        }
    return state


def load_trainable_state(model, state, lora_r):
    if state.get("lora_enabled") and not model.lora_enabled:
        model.enable_lora(lora_r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0)
    if "encoder_state" in state:
        model.encoder.load_state_dict(state["encoder_state"])
    proj = dict(state["projector_state"])
    cur = model.projector.state_dict()
    for k, v in proj.items():  # merged tokens: start as "average the group, then old projector"
        if k in cur and v.shape != cur[k].shape and v.dim() >= 1 and cur[k].shape[-1] % v.shape[-1] == 0:
            m = cur[k].shape[-1] // v.shape[-1]
            proj[k] = v.repeat(*([1] * (v.dim() - 1)), m) / (m if v.dim() == 2 else 1)
    model.projector.load_state_dict(proj)
    if state.get("lora_enabled"):
        params = dict(model.llm.named_parameters())
        for n, v in state["lora_state"].items():
            params[n].data.copy_(v.to(params[n].device))


def make_optimizer(model):
    groups = [{"params": list(model.projector.parameters()), "lr": LR_PROJECTOR, "weight_decay": WEIGHT_DECAY}]
    if model.encoder_trainable:
        lr = LR_ENCODER if model.ts_encoder == "cnn" else LR_ENCODER / 10  # gentle on a pretrained encoder
        groups.append({"params": list(model.encoder.parameters()), "lr": lr, "weight_decay": WEIGHT_DECAY})
    if model.lora_enabled:
        groups.append(
            {"params": model.get_lora_parameters(), "lr": LR_PROJECTOR, "weight_decay": WEIGHT_DECAY}
        )
    return AdamW(groups)


def answer_token_budget(model, dataset, n=4000, seed=0):
    rng = random.Random(seed)
    idxs = rng.sample(range(len(dataset)), min(n, len(dataset)))
    answers = [dataset[i]["answer"] for i in idxs]
    lens = sorted(len(x) for x in model.tokenizer(answers, add_special_tokens=False).input_ids)
    p999 = lens[min(len(lens) - 1, int(0.999 * len(lens)))]
    return int(min(1024, p999 * 1.25 + 16)), lens[len(lens) // 2], lens[-1]


@torch.no_grad()
def validate(model, dataset, bs):
    model.eval()
    idx = list(range(len(dataset)))[RANK::WORLD]
    total, n = 0.0, 0
    for b in range(0, len(idx), bs):
        total += model.compute_loss(collate([dataset[i] for i in idx[b : b + bs]])).item()
        n += 1
    if WORLD > 1:
        t = torch.tensor([total, n], device=f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu",
                         dtype=torch.float64)
        dist.all_reduce(t)
        total, n = t.tolist()
    model.train()
    return total / max(n, 1)


@torch.no_grad()
def run_test(model, dataset, stage, out_dir, eval_bs, max_new_tokens, limit=None, fname="test_predictions.jsonl"):
    force = FORCE_ANSWER and STAGE_DEFAULTS[stage]["task"] in ("har", "sleep", "ecg")
    model.eval()
    idxs = list(range(len(dataset)))
    if limit:
        idxs = random.Random(0).sample(idxs, min(limit, len(idxs)))
    # sort by prompt size (number of time-series points + text length), longest first
    def size(i):
        s = dataset[i]
        return sum(len(t) for t in s["time_series"]) + len(s["pre_prompt"]) + len(s["post_prompt"])

    idxs.sort(key=size, reverse=True)
    idxs = idxs[RANK::WORLD]  # strided shard keeps the length distribution balanced
    final_path = os.path.join(out_dir, fname)
    preds_path = final_path if WORLD == 1 else f"{final_path}.r{RANK}"
    done = 0
    t0 = time.time()
    gen_sub = eval_bs  # shrinks permanently after an OOM
    with open(preds_path, "w", encoding="utf-8") as f:
        for b in range(0, len(idxs), eval_bs):
            chunk = idxs[b : b + eval_bs]
            batch = collate([dict(dataset[i]) for i in chunk])
            preds = []
            while len(preds) < len(batch):  # halve the sub-batch on OOM instead of crashing
                try:
                    preds += model.generate(batch[len(preds) : len(preds) + gen_sub], max_new_tokens=max_new_tokens,
                                            do_sample=False)
                except torch.cuda.OutOfMemoryError:
                    if gen_sub == 1:
                        raise
                    torch.cuda.empty_cache()
                    gen_sub = max(1, gen_sub // 2)
                    log(out_dir, event="gen_oom", sub_batch=gen_sub)
            if force:
                missing = [j for j, pr in enumerate(preds) if "Answer:" not in pr]
                if missing:
                    cont_batch = []
                    for j in missing:
                        smp = dict(batch[j])
                        smp["post_prompt"] = smp["post_prompt"] + preds[j].rstrip() + " Answer:"
                        cont_batch.append(smp)
                    conts = model.generate(cont_batch, max_new_tokens=48, do_sample=False)
                    for j, c in zip(missing, conts):
                        preds[j] = preds[j].rstrip() + " Answer:" + c.split("\n")[0]
            for i, sample, pred in zip(chunk, batch, preds):
                rec = {
                    "idx": i,
                    "pre_prompt": sample["pre_prompt"],
                    "time_series_text": sample["time_series_text"],
                    "post_prompt": sample["post_prompt"],
                    "generated": pred,
                    "gold": sample["answer"],
                }
                for k in ("template_id", "ecg_id", "label"):
                    if k in sample and sample[k] is not None:
                        rec[k] = sample[k]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done += len(batch)
            if (b // eval_bs) % 20 == 0 and fname.startswith("test"):
                log(out_dir, event="test_progress", stage=stage, done=done * WORLD, total=len(idxs) * WORLD,
                    sec=round(time.time() - t0, 1))
    model.train()
    if WORLD == 1:
        return preds_path
    barrier()
    if RANK != 0:
        return None
    with open(final_path, "w", encoding="utf-8") as out:
        for r in range(WORLD):
            part = f"{final_path}.r{r}"
            with open(part, encoding="utf-8") as f:
                out.write(f.read())
            os.remove(part)
    return final_path


def run_stage(model, stage, args, root):
    import copy

    args = copy.copy(args)
    for k, v in args.overrides.get(stage, {}).items():
        setattr(args, k, v)
    if stage in args.overrides:
        log(root, event="stage_overrides", stage=stage, **args.overrides[stage])
    OPTS["sleep_uv"] = args.sleep_uv
    cfg = STAGE_DEFAULTS[stage]
    stage_dir = os.path.join(root, stage)
    os.makedirs(stage_dir, exist_ok=True)
    metrics_path = os.path.join(stage_dir, "metrics.json")
    best_path = os.path.join(stage_dir, "best.pt")
    last_path = os.path.join(stage_dir, "last.pt")

    if os.path.exists(metrics_path) and os.path.exists(best_path):
        load_trainable_state(model, torch.load(best_path, weights_only=False), args.lora_r)
        log(root, event="stage_skip_done", stage=stage)
        return json.load(open(metrics_path))

    if cfg["lora"] and not model.lora_enabled:
        model.enable_lora(lora_r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0)

    cls = dataset_class(stage)
    eos = model.get_eos_token()
    t_load = time.time()
    import contextlib

    quiet = stage == "stage5_ecg_cot" and OPTS.get("ecg_hz", 100) == 500
    parallel = stage == "stage5_ecg_cot"  # downloads are separable: rank 0 fetches, then all ranks build
    if WORLD > 1 and parallel:
        if RANK == 0:
            ensure_ecg_downloaded()
        barrier()
    elif WORLD > 1 and RANK != 0:
        barrier()  # wait until rank 0 has downloaded/extracted the dataset
    with (contextlib.redirect_stdout(open(os.devnull, "w")) if quiet else contextlib.nullcontext()):
        train_ds, val_ds, test_ds = cls("train", EOS_TOKEN=eos), cls("validation", EOS_TOKEN=eos), cls("test", EOS_TOKEN=eos)
    if WORLD > 1 and RANK == 0 and not parallel:
        barrier()
    log(root, event="data_loaded", stage=stage, train=len(train_ds), val=len(val_ds), test=len(test_ds),
        sec=round(time.time() - t_load, 1))
    if stage == "stage5_ecg_cot":  # formatted samples hold their own copies; drop the raw signal caches
        cls._ecg_data_cache.clear()
        cls._processed_ecg_cache.clear()
        getattr(cls, "_prompt_cache", {}).clear()

    if stage == "stage5_ecg_cot" and args.ecg_train_fraction < 1.0:
        keep = stratified_indices(train_ds, args.ecg_train_fraction, args.seed, args.ecg_subset_part)
        train_ds = Subset(train_ds, keep)
        log(root, event="ecg_subsample", kept=len(keep), fraction=args.ecg_train_fraction, part=args.ecg_subset_part)

    if args.require_answer_tag and cfg["task"] in ("har", "sleep", "ecg"):
        base, idx = (train_ds.dataset, train_ds.indices) if isinstance(train_ds, Subset) else (train_ds, range(len(train_ds)))
        good = [i for i in idx if "Answer:" in base[i]["answer"]]
        log(root, event="filter_answer_tag", kept=len(good), dropped=len(idx) - len(good))
        train_ds = Subset(base, good)

    val_idx = random.Random(args.seed).sample(range(len(val_ds)), min(args.val_samples, len(val_ds)))
    val_sub = Subset(val_ds, val_idx)

    bs = args.ecg_batch_size if stage == "stage5_ecg_cot" else args.batch_size
    accum = args.grad_accum
    initial_state = trainable_state(model)
    base_train = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
    select_metric = args.select_metric if cfg["task"] else None
    gen_budget = answer_token_budget(model, base_train)[0] if select_metric else None
    sample_weights = None
    if args.balance_power > 0:
        labels = [final_label(train_ds[i]["answer"]) for i in range(len(train_ds))]
        counts = defaultdict(int)
        for lab in labels:
            counts[lab] += 1
        # labels seen < 20 times are malformed rationales, not classes: give them the average weight
        real = {lab: c for lab, c in counts.items() if c >= 20}
        avg_w = sum(c ** (-args.balance_power) * c for c in real.values()) / sum(real.values())
        sample_weights = [counts[lab] ** (-args.balance_power) if lab in real else avg_w for lab in labels]
        log(root, event="balanced_sampling", power=args.balance_power, classes=real,
            n_unparsed=sum(c for lab, c in counts.items() if lab not in real))

    def train(bs, accum):
        epochs = args.epochs.get(stage, cfg["epochs"])
        n_train = len(train_ds)
        per_rank = n_train // WORLD
        n_batches = per_rank // bs

        def epoch_batches(epoch):
            # same permutation on every rank, each rank takes its own contiguous slice
            g = torch.Generator()
            g.manual_seed(args.seed + epoch)
            if sample_weights is not None:
                idx = torch.multinomial(torch.tensor(sample_weights, dtype=torch.double), n_train,
                                        replacement=True, generator=g).tolist()
            else:
                idx = torch.randperm(n_train, generator=g).tolist()
            idx = idx[RANK * per_rank : (RANK + 1) * per_rank]
            return [idx[i * bs : (i + 1) * bs] for i in range(n_batches)]

        steps_per_epoch = n_batches // accum
        total_steps = epochs * steps_per_epoch
        if args.max_steps or args.limit_steps:
            total_steps = min(total_steps, args.max_steps or args.limit_steps)
        evals_per_epoch = args.ecg_evals_per_epoch if stage == "stage5_ecg_cot" else 1
        eval_every = max(1, steps_per_epoch // evals_per_epoch)

        optimizer = make_optimizer(model)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_FRAC * total_steps), total_steps)

        step, best_sel, no_improve, start_epoch, skip_batches = 0, float("-inf"), 0, 0, 0
        if os.path.exists(last_path) and not args.max_steps:
            ck = torch.load(last_path, weights_only=False)
            load_trainable_state(model, ck["model"], args.lora_r)
            optimizer = make_optimizer(model)
            optimizer.load_state_dict(ck["optimizer"])
            scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_FRAC * total_steps), total_steps)
            scheduler.load_state_dict(ck["scheduler"])
            step, no_improve = ck["step"], ck["no_improve"]
            best_sel = ck["best_sel"] if "best_sel" in ck else -ck["best_val"]
            start_epoch, skip_batches = ck["epoch"], ck["batch_in_epoch"]
            log(root, event="resume", stage=stage, step=step, epoch=start_epoch, best_sel=best_sel)

        log(root, event="stage_start", stage=stage, epochs=epochs, batch_size=bs, grad_accum=accum, world=WORLD,
            steps_per_epoch=steps_per_epoch, total_steps=total_steps, eval_every=eval_every,
            trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
            attn=model.llm.config._attn_implementation, compile=args.compile)
        trainable = [p for p in model.parameters() if p.requires_grad]
        if WORLD > 1:
            for p in trainable:
                dist.broadcast(p.data, 0)
        if args.compile:
            base = model.llm.get_base_model() if model.lora_enabled else model.llm
            object.__setattr__(model, "compiled_body", torch.compile(base.model, dynamic=True))

        model.train()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        done, t_win, tok_win = False, time.time(), 0
        for epoch in range(start_epoch, epochs):
            for bi, bidx in enumerate(epoch_batches(epoch)):
                if epoch == start_epoch and bi < skip_batches:
                    continue
                batch = collate([train_ds[i] for i in bidx])
                loss = model.compute_loss(batch) / accum
                loss.backward()
                tok_win += model.last_num_tokens * WORLD
                if (bi + 1) % accum:
                    continue
                sync_grads(trainable)
                clip_grad_norm_(trainable, GRAD_CLIP_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % args.log_every == 0:
                    dt = time.time() - t_win
                    log(root, event="train", stage=stage, step=step, epoch=epoch,
                        loss=round(loss.item() * accum, 4), lr=scheduler.get_last_lr()[0],
                        tok_per_s=round(tok_win / dt), step_s=round(dt / args.log_every, 3),
                        peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1) if torch.cuda.is_available() else 0)
                    t_win, tok_win = time.time(), 0

                if args.max_steps and step >= total_steps:  # probe: no validation
                    done = True
                    break
                if step % eval_every == 0 or step >= total_steps:
                    vl = validate(model, val_sub, bs)
                    sel, extra = -vl, {}
                    if select_metric:
                        vp = run_test(model, val_ds, stage, stage_dir, args.eval_batch_size, gen_budget,
                                      args.val_gen_samples, fname="val_predictions.jsonl")
                        vm = score.score(cfg["task"], vp) if RANK == 0 else {}
                        sel = bcast(vm.get(select_metric))
                        extra = {f"val_{k}": round(v, 2) for k, v in vm.items() if isinstance(v, (int, float))}
                    improved = sel > best_sel + (1e-6 if select_metric else 1e-4)
                    if improved:
                        best_sel, no_improve = sel, 0
                        if RANK == 0:
                            torch.save(trainable_state(model), best_path)
                    else:
                        no_improve += 1
                    if RANK == 0:
                        torch.save({"model": trainable_state(model), "optimizer": optimizer.state_dict(),
                                    "scheduler": scheduler.state_dict(), "step": step, "best_sel": best_sel,
                                    "no_improve": no_improve, "epoch": epoch, "batch_in_epoch": bi + 1}, last_path)
                    barrier()
                    log(root, event="val", stage=stage, step=step, epoch=epoch, val_loss=round(vl, 4),
                        selection=select_metric or "neg_val_loss", best_sel=round(best_sel, 4),
                        no_improve=no_improve, **extra)
                    if no_improve >= args.patience:
                        log(root, event="early_stop", stage=stage, step=step)
                        done = True
                        break
                    if step >= total_steps:
                        done = True
                        break
            if done:
                break
        return best_sel, step

    while not args.eval_only:
        try:
            best_sel, step = train(bs, accum)
            break
        except torch.cuda.OutOfMemoryError:
            if bs == 1 or WORLD > 1:  # ranks cannot retry independently
                raise
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            for pth in (best_path, last_path):
                if os.path.exists(pth):
                    os.remove(pth)
            load_trainable_state(model, initial_state, args.lora_r)
            model.zero_grad(set_to_none=True)
            bs, accum = bs // 2, accum * 2
            log(root, event="oom_retry", stage=stage, batch_size=bs, grad_accum=accum)

    if args.eval_only:  # test the --init_from weights as they are
        if RANK == 0:
            torch.save(trainable_state(model), best_path)
        best_sel, step = None, 0
    barrier()
    if args.max_steps:  # throughput probe only
        log(root, event="probe_done", stage=stage)
        return None

    load_trainable_state(model, torch.load(best_path, weights_only=False), args.lora_r)

    metrics = {"stage": stage, "selection": select_metric or "neg_val_loss", "best_selection": best_sel, "steps": step}
    eval_bs = args.ecg_eval_batch_size if stage == "stage5_ecg_cot" else args.eval_batch_size
    if cfg["task"] and (not args.skip_test or args.hf_compare or args.dump_test):
        budget, med, mx = answer_token_budget(model, train_ds.dataset if isinstance(train_ds, Subset) else train_ds)
        log(root, event="gen_budget", stage=stage, max_new_tokens=budget, median_answer_tokens=med, max_answer_tokens=mx)
    if cfg["task"] and args.dump_test and RANK == 0:  # test set for vllm_eval.py, so it need not rebuild it
        os.makedirs(args.dump_test, exist_ok=True)
        t0 = time.time()
        with open(os.path.join(args.dump_test, f"{stage}.pkl"), "wb") as f:
            pickle.dump({"samples": list(test_ds.dataset), "budget": budget}, f, protocol=5)
        log(root, event="test_dumped", stage=stage, n=len(test_ds), sec=round(time.time() - t0, 1))
    if cfg["task"] and args.skip_test and args.hf_compare:  # HF generations on a subset, to check vLLM against
        t0 = time.time()
        preds = run_test(model, test_ds, stage, stage_dir, eval_bs, budget, args.hf_compare,
                         fname="test_predictions_hf.jsonl")
        metrics["hf_compare_seconds"] = round(time.time() - t0)
        if RANK == 0:
            metrics["hf_compare"] = {"n": args.hf_compare, **score.score(cfg["task"], preds)}
    if cfg["task"] and not args.skip_test:
        t0 = time.time()
        preds = run_test(model, test_ds, stage, stage_dir, eval_bs, budget, args.test_samples)
        metrics["test_seconds"] = round(time.time() - t0)
        if RANK == 0:
            try:
                metrics.update(score.score(cfg["task"], preds))
            except Exception as e:  # predictions are on disk; scoring can be rerun
                metrics["score_error"] = repr(e)
    if RANK == 0:
        json.dump(metrics, open(metrics_path, "w"), indent=2)
        if os.path.exists(last_path):
            os.remove(last_path)
    log(root, event="stage_done", **metrics)
    barrier()
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_id", default="LiquidAI/LFM3-1.7B-Base")
    ap.add_argument("--chat_template", action="store_true")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--stages", nargs="+", default=STAGE_ORDER)
    ap.add_argument("--init_from", default=None, help="best.pt of a previous stage to start from")
    ap.add_argument("--epochs", default="", help="e.g. stage1_mcq=3,stage4_sleep_cot=10")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--ecg_batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--ecg_eval_batch_size", type=int, default=32)
    ap.add_argument("--val_samples", type=int, default=1000)
    ap.add_argument("--ecg_train_fraction", type=float, default=0.25)
    ap.add_argument("--ecg_evals_per_epoch", type=int, default=4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--test_samples", type=int, default=None)
    ap.add_argument("--skip_test", action="store_true")
    ap.add_argument("--max_steps", type=int, default=None, help="throughput probe: train N steps, no eval")
    ap.add_argument("--limit_steps", type=int, default=None, help="debug: cap steps per stage, still eval")
    ap.add_argument("--max_patches", type=int, default=4096, help="encoder positional table (M4 has ~10k-point series)")
    ap.add_argument("--ts_encoder", default="cnn", choices=["cnn", "chronos2"])
    ap.add_argument("--ecg_hz", type=int, default=100, choices=[100, 500], help="ECG sampling rate fed to the encoder")
    ap.add_argument("--train_encoder", action="store_true", help="also train a pretrained (chronos2) encoder")
    ap.add_argument("--answer_weight", type=float, default=1.0, help="loss weight of tokens after the final 'Answer:'")
    ap.add_argument("--force_answer", action="store_true", help="finish generations that lack 'Answer:'")
    ap.add_argument("--overrides", default="{}", help='per-stage arg overrides as JSON, e.g. {"stage3_cot": {"balance_power": 0.5}}')
    ap.add_argument("--eval_only", action="store_true", help="skip training; evaluate --init_from on the test set")
    ap.add_argument("--ecg_subset_part", type=int, default=0, help="k-th disjoint stratified ECG subset")
    ap.add_argument("--require_answer_tag", action="store_true", help="drop CoT training targets without 'Answer:'")
    ap.add_argument("--select_metric", default=None, help="pick checkpoints by a generated val metric, e.g. macro_f1")
    ap.add_argument("--val_gen_samples", type=int, default=None, help="val samples generated for --select_metric")
    ap.add_argument("--balance_power", type=float, default=0.0, help="class-balanced sampling: weight = count^-p")
    ap.add_argument("--sleep_uv", action="store_true", help="state EEG mean/std in microvolts in the Sleep prompt")
    ap.add_argument("--attn", default="sdpa", help="sdpa | flash_attention_2 | auto (flash if installed)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the LLM body for training")
    ap.add_argument("--merge_tokens", type=int, default=1, help="Chronos-2 patch tokens merged per LLM token")
    ap.add_argument("--export_dir", default=None, help="after training: save LoRA-merged LLM + embedder for vLLM")
    ap.add_argument("--dump_test", default=None, help="dir: pickle each stage's test set + token budget for vllm_eval.py")
    ap.add_argument("--hf_compare", type=int, default=0, help="with --skip_test: HF-generate N test samples anyway")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.epochs = {k: int(v) for k, v in (kv.split("=") for kv in args.epochs.split(",") if kv)}

    global FORCE_ANSWER, RANK, WORLD
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        import datetime

        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                                timeout=datetime.timedelta(hours=3))  # dataset loading/eval run on rank 0 alone
        RANK, WORLD = dist.get_rank(), dist.get_world_size()
        if torch.cuda.is_available():
            local = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local)
            args.device = f"cuda:{local}"
    if args.attn == "auto":
        try:
            import flash_attn  # noqa: F401

            args.attn = "flash_attention_2" if torch.cuda.is_available() else "sdpa"
        except Exception:
            args.attn = "sdpa"
    args.overrides = json.load(open(args.overrides[1:])) if args.overrides.startswith("@") else json.loads(args.overrides)
    OPTS["sleep_uv"] = args.sleep_uv
    OPTS["collate_patch"] = 1 if args.ts_encoder == "chronos2" else PATCH_SIZE
    OPTS["ecg_hz"] = args.ecg_hz
    FORCE_ANSWER = args.force_answer
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out_dir, exist_ok=True)
    if RANK == 0:
        json.dump(vars(args), open(os.path.join(args.out_dir, "args.json"), "w"), indent=2)

    model = OpenTSLMSP(llm_id=args.llm_id, device=args.device, attn_implementation=args.attn,
                       chat_template=args.chat_template, max_patches=args.max_patches, ts_encoder=args.ts_encoder,
                       encoder_trainable=args.ts_encoder == "cnn" or args.train_encoder,
                       merge_tokens=args.merge_tokens)
    model.answer_weight = args.answer_weight
    if args.init_from:
        load_trainable_state(model, torch.load(args.init_from, weights_only=False), args.lora_r)
    log(args.out_dir, event="model_loaded", llm_id=args.llm_id, chat_template=args.chat_template,
        eos=model.get_eos_token(), bos=model.tokenizer.bos_token)

    results = {}
    for stage in args.stages:
        results[stage] = run_stage(model, stage, args, args.out_dir)
    if RANK == 0:
        json.dump(results, open(os.path.join(args.out_dir, "results.json"), "w"), indent=2)
        if args.export_dir:
            export_for_vllm(model, args)
    barrier()
    log(args.out_dir, event="all_done")
    if WORLD > 1:
        dist.destroy_process_group()


def export_for_vllm(model, args):
    """LoRA-merged LLM (HF format) + the time-series embedder state, for scripts/lfm3/vllm_eval.py."""
    os.makedirs(args.export_dir, exist_ok=True)
    state = trainable_state(model)
    llm = model.llm.merge_and_unload() if model.lora_enabled else model.llm
    llm.save_pretrained(os.path.join(args.export_dir, "llm"), safe_serialization=True)
    model.tokenizer.save_pretrained(os.path.join(args.export_dir, "llm"))
    torch.save({"projector_state": state["projector_state"], "encoder_state": state.get("encoder_state"),
                "embed_tokens": llm.get_input_embeddings().weight.detach().cpu(),
                "config": {"ts_encoder": model.ts_encoder, "merge_tokens": model.merge_tokens,
                           "ecg_hz": OPTS.get("ecg_hz", 100), "sleep_uv": OPTS.get("sleep_uv", False),
                           "llm_id": args.llm_id, "hidden_size": llm.config.hidden_size}},
               os.path.join(args.export_dir, "embedder.pt"))
    log(args.out_dir, event="exported", dir=args.export_dir)


if __name__ == "__main__":
    main()
