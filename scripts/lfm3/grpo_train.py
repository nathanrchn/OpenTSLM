# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT
"""
GRPO fine-tuning of an exported OpenTSLM-SoftPrompt model (train_sp.py --export_dir) on a chain-of-thought task
with verifiable labels (ECG-QA-CoT by default).

- Policy: the SFT model with its LoRA already merged (<export>/llm) plus a fresh LoRA. The Chronos-2 encoder
  and the projector stay frozen, so prompt embeddings are the SFT ones. The reference policy for the optional
  KL term is the same model with the adapter disabled, i.e. exactly the SFT model.
- Rollouts: vLLM runs inside each training process (one engine per GPU) and samples from the policy's own
  prompt embeddings (`prompt_embeds`). After every update the merged weights W + scale*B@A of the LoRA layers
  are loaded into the engine in place (computed out of place: no rounding drift in W). With --vllm_sleep the
  engine frees its memory while the policy trains.
- Objective: GRPO with Dr. GRPO normalisation (advantage = reward - group mean, no std scaling; per-sequence
  token sums divided by a constant), one on-policy update per batch, optional truncated importance weights
  for the vLLM-vs-HF numeric mismatch (--tis_cap), optional k3 KL to the SFT model (--kl_beta).
- Reward: 1 if the final "Answer:" equals the label (the paper scorer's normalisation), times the chance
  weight 1 - 1/n_options (--reward_weight chance): a lucky guess among 3 options earns less than a correct
  pick among 40.

Run on GPUs with torchrun (vLLM env, peft installed); --engine hf runs the same loop with HF sampling
(slow, for CPU tests).
"""

import argparse
import contextlib
import json
import math
import os
import pickle
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_sp  # noqa: E402  (datasets, collate, budget, scorer, distributed helpers)
import vllm_eval  # noqa: E402  (record format)

RANK, WORLD, DEV = 0, 1, "cpu"


def log(out_dir, **kw):
    train_sp.log(out_dir, **kw)


def allreduce_sum(vals):
    if WORLD == 1:
        return list(vals)
    t = torch.tensor(vals, dtype=torch.float64, device=DEV)
    dist.all_reduce(t)
    return t.tolist()


def answer_of(text: str) -> Optional[str]:
    """Final answer after the last "Answer:", normalised like the paper's ECG/HAR/Sleep scorers."""
    if "Answer:" not in text:
        return None
    a = re.sub(r"<\|.*?\|>|<eos>", "", text.split("Answer:")[-1])
    return a.strip().lower().rstrip(".,!?;:").strip()


@dataclass
class Completion:
    ids: List[int]
    text: str
    logprobs: Optional[List[Optional[float]]]  # rollout-engine logprob of each token (None if unknown)


# ----------------------------------------------------------------------------------------------- policy


def build_policy(args):
    from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP

    blob = torch.load(os.path.join(args.export_dir, "embedder.pt"), weights_only=False, map_location="cpu")
    cfg = blob["config"]
    train_sp.OPTS.update({"collate_patch": 1, "ecg_hz": cfg["ecg_hz"], "sleep_uv": cfg["sleep_uv"]})
    with contextlib.redirect_stdout(open(os.devnull, "w")) if RANK else contextlib.nullcontext():
        model = OpenTSLMSP(llm_id=os.path.join(args.export_dir, "llm"), device=DEV, attn_implementation="sdpa",
                           ts_encoder=cfg["ts_encoder"], encoder_trainable=False,
                           merge_tokens=cfg["merge_tokens"])
        if blob.get("encoder_state"):
            model.encoder.load_state_dict(blob["encoder_state"])
        model.projector.load_state_dict(blob["projector_state"])
        model.encoder.requires_grad_(False)
        model.projector.requires_grad_(False)
        model.enable_lora(lora_r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0)
    model.eval()  # no dropout anywhere; gradients still flow into the LoRA weights
    return model


def lora_params(model):
    return [p for n, p in model.llm.named_parameters() if p.requires_grad and "lora_" in n]


def merged_lora_weights(model):
    """Yields (HF checkpoint name, W + delta) for every LoRA-wrapped linear layer, one layer at a time (low
    peak memory); W itself is left untouched, so repeated syncs accumulate no rounding error."""
    from peft.tuners.lora import LoraLayer

    for name, mod in model.llm.get_base_model().named_modules():
        if isinstance(mod, LoraLayer) and "default" in mod.lora_A:
            with torch.no_grad():
                w = mod.get_base_layer().weight
                yield f"{name}.weight", (w.float() + mod.get_delta_weight("default").float()).to(w.dtype)


def policy_logprobs(model, prompts, comps):
    """Log-probability of each completion token under the policy. prompts: [L_i, H] embeddings (no grad);
    comps: token id lists. Logits only at completion positions (as in OpenTSLMSP.compute_loss)."""
    emb = model.llm.get_input_embeddings()
    seqs, spans = [], []
    for p, ids in zip(prompts, comps):
        t = torch.tensor(ids, device=DEV, dtype=torch.long)
        seqs.append(torch.cat([p, emb(t).to(p.dtype)], dim=0))
        spans.append((p.size(0), len(ids)))
    x = pad_sequence(seqs, batch_first=True)
    am = pad_sequence([torch.ones(s.size(0), dtype=torch.long, device=DEV) for s in seqs], batch_first=True)
    base = model.llm.get_base_model()
    h = base.model(inputs_embeds=x, attention_mask=am).last_hidden_state
    rows = torch.cat([h[i, P - 1 : P - 1 + n] for i, (P, n) in enumerate(spans)])  # position t predicts t+1
    tgt = torch.cat([torch.tensor(c, device=DEV, dtype=torch.long) for c in comps])
    lp = -torch.nn.functional.cross_entropy(base.lm_head(rows).float(), tgt, reduction="none")
    return list(lp.split([n for _, n in spans]))


# ----------------------------------------------------------------------------------------------- engines


class HFEngine:
    """Rollouts with HF generate on the policy itself (slow; CPU tests and fallback)."""

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def generate(self, prompts, n, temperature, max_tokens, eos, want_logprobs=False):
        groups = []
        for e in prompts:
            x = e.unsqueeze(0)
            kw = dict(do_sample=True, temperature=temperature, top_p=1.0) if temperature > 0 else dict(do_sample=False)
            out = self.model.llm.generate(inputs_embeds=x, attention_mask=torch.ones(x.shape[:2], dtype=torch.long,
                                          device=DEV), max_new_tokens=max_tokens, num_return_sequences=n,
                                          eos_token_id=eos, pad_token_id=eos, **kw)
            group = []
            for row in out.tolist():
                ids = []
                for t in row:
                    ids.append(t)
                    if t == eos:
                        break
                group.append(Completion(ids, self.model.tokenizer.decode(ids, skip_special_tokens=True), None))
            groups.append(group)
        return groups

    def load(self, weights):
        pass

    def sleep(self):
        pass

    def wake(self):
        pass


class VLLMEngine:
    """vLLM in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0), one engine per rank (TP=1)."""

    def __init__(self, llm_dir, args):
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        from vllm import LLM, SamplingParams

        self.SamplingParams = SamplingParams
        kw = dict(model=llm_dir, dtype="bfloat16", enable_prompt_embeds=True, trust_remote_code=True,
                  gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len, seed=args.seed + RANK,
                  enable_prefix_caching=False, enable_sleep_mode=bool(args.vllm_sleep))
        if WORLD > 1:
            kw["distributed_executor_backend"] = "external_launcher"
        self.llm = LLM(**kw)
        self.sleep_mode = bool(args.vllm_sleep)
        self.awake = True

    def generate(self, prompts, n, temperature, max_tokens, eos, want_logprobs=False):
        sp = self.SamplingParams(n=n, temperature=temperature, top_p=1.0, max_tokens=max_tokens,
                                 stop_token_ids=[eos], skip_special_tokens=True,
                                 logprobs=0 if want_logprobs else None)
        outs = self.llm.generate([{"prompt_embeds": e.detach().to("cpu")} for e in prompts], sp, use_tqdm=False)
        groups = []
        for o in outs:
            group = []
            for c in o.outputs:
                ids = list(c.token_ids)
                lps = None
                if want_logprobs and c.logprobs is not None:
                    lps = [d[t].logprob if (d is not None and t in d) else None for d, t in zip(c.logprobs, ids)]
                if c.finish_reason == "stop" and (not ids or ids[-1] != eos):  # learn to stop as well
                    ids.append(eos)
                    if lps is not None:
                        lps.append(None)
                group.append(Completion(ids, c.text, lps))
            groups.append(group)
        return groups

    def _model(self):
        eng = self.llm.llm_engine
        for path in ("model_executor.driver_worker.worker.model_runner.model",
                     "model_executor.driver_worker.model_runner.model",
                     "engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model",
                     "engine_core.engine_core.model_executor.driver_worker.model_runner.model"):
            obj = eng
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        return None

    @torch.no_grad()
    def load(self, weights):
        model = self._model()
        if model is not None:
            model.load_weights(weights=weights)
        else:  # public API fallback (in-process engine: the closure's GPU tensors are passed by reference)
            self.llm.apply_model(lambda m: m.load_weights(weights=weights))
        with contextlib.suppress(Exception):
            self.llm.reset_prefix_cache()

    def sleep(self):
        if self.sleep_mode and self.awake:
            self.llm.sleep(level=1)
            self.awake = False

    def wake(self):
        if self.sleep_mode and not self.awake:
            self.llm.wake_up()
            self.awake = True


# ----------------------------------------------------------------------------------------------- data


def load_data(args, model):
    cls = train_sp.dataset_class(args.stage)
    if args.stage == "stage5_ecg_cot":
        if RANK == 0:
            train_sp.ensure_ecg_downloaded()
        train_sp.barrier()
    elif RANK != 0:
        train_sp.barrier()
    eos = model.get_eos_token()
    quiet = args.stage == "stage5_ecg_cot"
    t0 = time.time()
    with contextlib.redirect_stdout(open(os.devnull, "w")) if quiet else contextlib.nullcontext():
        train_ds, val_ds, test_ds = cls("train", EOS_TOKEN=eos), cls("validation", EOS_TOKEN=eos), cls("test", EOS_TOKEN=eos)
    if args.stage != "stage5_ecg_cot" and RANK == 0:
        train_sp.barrier()
    if args.stage == "stage5_ecg_cot":
        cls._ecg_data_cache.clear()
        cls._processed_ecg_cache.clear()
        getattr(cls, "_prompt_cache", {}).clear()
    pool = list(range(len(train_ds)))
    if args.stage == "stage5_ecg_cot" and args.ecg_train_fraction < 1.0:  # the half SFT did not train on
        pool = train_sp.stratified_indices(train_ds, args.ecg_train_fraction, args.seed, args.ecg_subset_part)
    pool = [i for i in pool if answer_of(train_ds[i]["answer"])]
    budget = train_sp.answer_token_budget(model, train_ds)[0]
    log(args.out_dir, event="data_loaded", train=len(train_ds), pool=len(pool), val=len(val_ds), test=len(test_ds),
        budget=budget, sec=round(time.time() - t0, 1))
    if args.dump_test and RANK == 0:  # test set for vllm_eval.py (no dataset rebuild there)
        os.makedirs(args.dump_test, exist_ok=True)
        with open(os.path.join(args.dump_test, f"{args.stage}.pkl"), "wb") as f:
            pickle.dump({"samples": list(test_ds.dataset), "budget": budget}, f, protocol=5)
    return cls, train_ds, val_ds, pool, budget


def n_options(cls, stage, sample):
    if stage == "stage5_ecg_cot" and sample.get("template_id") is not None:
        n = len(cls.get_possible_answers_for_template(sample["template_id"]))
        if n:
            return n
    m = re.search(r"following options:\s*\n(.*?)\n", sample.get("post_prompt", ""))
    return max(2, len(m.group(1).split(", "))) if m else 2


# ----------------------------------------------------------------------------------------------- loop


def validate(args, model, engine, val_ds, val_idx, step, budget):
    """Greedy answers on the fixed validation subset (sharded over ranks), scored with the paper scorer."""
    task = train_sp.STAGE_DEFAULTS[args.stage]["task"]
    mine = val_idx[RANK::WORLD]
    path = os.path.join(args.out_dir, f"val_step{step}.jsonl")
    eos = model.tokenizer.eos_token_id
    t0 = time.time()
    with open(f"{path}.r{RANK}", "w", encoding="utf-8") as f:
        for b in range(0, len(mine), args.val_chunk):
            chunk = mine[b : b + args.val_chunk]
            batch = train_sp.collate([val_ds[i] for i in chunk])
            with torch.no_grad():
                prompts = model._build_prompt_embeds(batch)
            groups = engine.generate(prompts, 1, 0.0, budget, eos)
            for i, s, g in zip(chunk, batch, groups):
                f.write(json.dumps(vllm_eval.record(s, g[0].text, i), ensure_ascii=False) + "\n")
    train_sp.barrier()
    metrics = {}
    if RANK == 0:
        with open(path, "w", encoding="utf-8") as out:
            for r in range(WORLD):
                with open(f"{path}.r{r}", encoding="utf-8") as f:
                    out.write(f.read())
                os.remove(f"{path}.r{r}")
        metrics = train_sp.score.score(task, path)
    sel = train_sp.bcast(metrics.get(args.select_metric) if RANK == 0 else None)
    log(args.out_dir, event="val", step=step, sec=round(time.time() - t0, 1), selection=args.select_metric,
        **{f"val_{k}": round(v, 2) for k, v in metrics.items() if isinstance(v, (int, float))})
    return sel


def save_adapter(model, path, **meta):
    if RANK == 0:
        torch.save({"lora": {n: p.detach().cpu().clone() for n, p in model.llm.named_parameters() if "lora_" in n},
                    **meta}, path)


def export(model, args, best_path):
    """LoRA-merged LLM + the SFT embedder (encoder/projector were frozen) in vllm_eval.py's export format."""
    if RANK != 0:
        return
    if os.path.exists(best_path):
        state = torch.load(best_path, weights_only=False)["lora"]
        params = dict(model.llm.named_parameters())
        for n, v in state.items():
            params[n].data.copy_(v.to(params[n].device))
    out = os.path.join(args.export_out, "llm")
    os.makedirs(out, exist_ok=True)
    llm = model.llm.merge_and_unload()
    llm.save_pretrained(out, safe_serialization=True)
    model.tokenizer.save_pretrained(out)
    shutil.copy(os.path.join(args.export_dir, "embedder.pt"), os.path.join(args.export_out, "embedder.pt"))
    log(args.out_dir, event="exported", dir=args.export_out)


def train(args):
    global RANK, WORLD, DEV
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        import datetime

        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", timeout=datetime.timedelta(hours=2))
        RANK, WORLD = dist.get_rank(), dist.get_world_size()
    if torch.cuda.is_available():
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        DEV = f"cuda:{local}"
    train_sp.RANK, train_sp.WORLD = RANK, WORLD
    os.makedirs(args.out_dir, exist_ok=True)
    if RANK == 0:
        json.dump(vars(args), open(os.path.join(args.out_dir, "args.json"), "w"), indent=2)
    torch.manual_seed(args.seed)  # identical LoRA init on every rank (and broadcast below to be sure)
    model = build_policy(args)
    if WORLD > 1:
        for p in lora_params(model):
            dist.broadcast(p.data, 0)
    torch.manual_seed(args.seed + RANK)  # per-rank sampling streams from here on
    random.seed(args.seed + RANK)
    cls, train_ds, val_ds, pool, budget = load_data(args, model)
    budget = args.max_new_tokens or budget
    eos = model.tokenizer.eos_token_id
    params = lora_params(model)
    log(args.out_dir, event="policy_ready", lora_params=sum(p.numel() for p in params), world=WORLD, engine=args.engine)

    # prompt sampling: class-balanced over final labels (as in SFT), a different slice per rank and step
    labels = [answer_of(train_ds[i]["answer"]) for i in pool]
    counts = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    weights = torch.tensor([counts[lab] ** -args.balance_power for lab in labels], dtype=torch.double)
    chance = [1.0 - 1.0 / n_options(cls, args.stage, train_ds[i]) if args.reward_weight == "chance" else 1.0
              for i in pool]
    val_idx = random.Random(0).sample(range(len(val_ds)), min(args.val_samples, len(val_ds)))  # = SFT's subset

    engine = HFEngine(model) if args.engine == "hf" else VLLMEngine(os.path.join(args.export_dir, "llm"), args)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup)))
    best_path = os.path.join(args.out_dir, "best_adapter.pt")
    p_local = max(1, args.prompts_per_step // WORLD)
    t_start = time.time()

    best = validate(args, model, engine, val_ds, val_idx, 0, budget)  # the SFT model under this engine
    save_adapter(model, best_path, step=0, val=best)
    no_improve = 0
    for step in range(1, args.steps + 1):
        t0 = time.time()
        g = torch.Generator().manual_seed(args.seed * 100003 + step)
        pick = torch.multinomial(weights, p_local * WORLD, replacement=True, generator=g).tolist()
        pick = pick[RANK * p_local : (RANK + 1) * p_local]
        batch = train_sp.collate([train_ds[pool[k]] for k in pick])
        with torch.no_grad():
            prompts = [p.detach() for p in model._build_prompt_embeds(batch)]
        engine.wake()
        groups = engine.generate(prompts, args.group_size, args.temperature, budget, eos, want_logprobs=args.tis_cap > 0)
        engine.sleep()
        t_gen = time.time() - t0

        # rewards and group-relative advantages
        items, n_correct, n_tag, n_len, n_mixed, n_all_right, n_all_wrong = [], 0, 0, 0, 0, 0, 0
        for q, (k, group) in enumerate(zip(pick, groups)):
            gold = answer_of(batch[q]["answer"])
            ok = [answer_of(c.text) == gold for c in group]
            if os.environ.get("GRPO_DEBUG_RANDOM_REWARD"):  # CPU smoke tests: exercise the update path
                ok = [random.random() < 0.5 for _ in group]
            r = [chance[k] * float(o) for o in ok]
            mean = sum(r) / len(r)
            n_correct += sum(ok)
            n_tag += sum("Answer:" in c.text for c in group)
            n_len += sum(len(c.ids) for c in group)
            n_mixed += 0 < sum(ok) < len(ok)
            n_all_right += sum(ok) == len(ok)
            n_all_wrong += sum(ok) == 0
            for i, c in enumerate(group):
                adv = r[i] - mean
                if abs(adv) > 1e-8 or args.kl_beta > 0:
                    items.append((q, i, adv))

        # policy gradient (one on-policy update): loss = -sum_t A * w_t * ratio_t / L, ratio = exp(lp - lp.detach())
        t1 = time.time()
        norm = p_local * args.group_size * budget
        items.sort(key=lambda it: len(groups[it[0]][it[1]].ids))
        loss_sum, kl_sum, mismatch_sum, mismatch_n = 0.0, 0.0, 0.0, 0
        for b in range(0, len(items), args.micro_batch):
            mb = items[b : b + args.micro_batch]
            comps = [groups[q][i].ids for q, i, _ in mb]
            lps = policy_logprobs(model, [prompts[q] for q, _, _ in mb], comps)
            ref = None
            if args.kl_beta > 0:
                with torch.no_grad(), model.llm.disable_adapter():
                    ref = policy_logprobs(model, [prompts[q] for q, _, _ in mb], comps)
            total = 0.0
            for j, ((q, i, adv), lp) in enumerate(zip(mb, lps)):
                w = torch.ones_like(lp)
                old = groups[q][i].logprobs
                if old is not None:
                    known = torch.tensor([x is not None for x in old], device=DEV)
                    old_t = torch.tensor([x if x is not None else 0.0 for x in old], device=DEV)
                    diff = (lp.detach() - old_t)[known]
                    mismatch_sum += diff.abs().sum().item()
                    mismatch_n += int(known.sum())
                    if args.tis_cap > 0:  # truncated importance weight pi_theta / pi_vllm
                        w = torch.where(known, (lp.detach() - old_t).exp().clamp(max=args.tis_cap), w)
                ratio = (lp - lp.detach()).exp()
                term = -(adv * w * ratio).sum()
                if ref is not None:
                    d = ref[j] - lp
                    kl = (d.exp() - d - 1).sum()
                    term = term + args.kl_beta * kl
                    kl_sum += kl.item()
                total = total + term
            (total / norm).backward()
            loss_sum += total.item() / norm
        train_sp.sync_grads(params)
        gnorm = clip_grad_norm_(params, args.clip).item()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        t_train = time.time() - t1

        t2 = time.time()
        engine.wake()
        engine.load(merged_lora_weights(model))  # the next rollouts / validation use the updated policy
        t_sync = time.time() - t2

        n_q = p_local
        n_c = p_local * args.group_size
        s = allreduce_sum([n_correct, n_tag, n_len, n_mixed, n_all_right, n_all_wrong, len(items), loss_sum,
                           kl_sum, mismatch_sum, mismatch_n, n_q, n_c])
        log(args.out_dir, event="step", step=step, acc=round(100 * s[0] / s[12], 2), tag_rate=round(100 * s[1] / s[12], 2),
            mean_len=round(s[2] / s[12], 1), mixed=round(100 * s[3] / s[11], 1), all_right=round(100 * s[4] / s[11], 1),
            all_wrong=round(100 * s[5] / s[11], 1), trained_seqs=int(s[6]), loss=round(s[7] / WORLD, 5),
            kl=round(s[8] / max(1, s[12]), 5), logp_mismatch=round(s[9] / s[10], 4) if s[10] else None,
            grad_norm=round(gnorm, 4), lr=sched.get_last_lr()[0], gen_s=round(t_gen, 1), train_s=round(t_train, 1),
            sync_s=round(t_sync, 1),
            peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1) if torch.cuda.is_available() else 0)

        if step % args.val_every == 0 or step == args.steps:
            sel = validate(args, model, engine, val_ds, val_idx, step, budget)
            if sel > best + 1e-6:
                best, no_improve = sel, 0
                save_adapter(model, best_path, step=step, val=sel)
            else:
                no_improve += 1
            log(args.out_dir, event="val_summary", step=step, best=round(best, 3), no_improve=no_improve)
            if no_improve >= args.patience:
                log(args.out_dir, event="early_stop", step=step)
                break
        if args.max_hours and time.time() - t_start > args.max_hours * 3600:
            log(args.out_dir, event="time_limit", step=step)
            break
    train_sp.barrier()
    if args.export_out:
        engine.sleep()
        del engine
        export(model, args, best_path)
    train_sp.barrier()
    log(args.out_dir, event="all_done", best=best)
    if WORLD > 1:
        dist.destroy_process_group()


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--export_dir", required=True, help="train_sp.py --export_dir output (SFT model)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--export_out", default=None, help="write the best policy here in export format")
    ap.add_argument("--dump_test", default=None, help="dir: pickle the test set + budget for vllm_eval.py")
    ap.add_argument("--stage", default="stage5_ecg_cot")
    ap.add_argument("--engine", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--prompts_per_step", type=int, default=64, help="questions per step over all ranks")
    ap.add_argument("--group_size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=None, help="default: the SFT answer budget")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--micro_batch", type=int, default=4, help="sequences per forward/backward")
    ap.add_argument("--kl_beta", type=float, default=0.0)
    ap.add_argument("--tis_cap", type=float, default=2.0, help="truncated importance weight cap (0: off)")
    ap.add_argument("--reward_weight", default="chance", choices=["chance", "none"])
    ap.add_argument("--balance_power", type=float, default=0.5)
    ap.add_argument("--ecg_train_fraction", type=float, default=0.5)
    ap.add_argument("--ecg_subset_part", type=int, default=1, help="1 = the half the SFT run did not use")
    ap.add_argument("--val_every", type=int, default=25)
    ap.add_argument("--val_samples", type=int, default=1500)
    ap.add_argument("--val_chunk", type=int, default=256)
    ap.add_argument("--select_metric", default="macro_f1_weighted")
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--gpu_mem", type=float, default=0.45, help="vLLM share of the GPU (the policy uses the rest)")
    ap.add_argument("--vllm_sleep", type=int, default=1, help="free vLLM memory while the policy trains")
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--max_hours", type=float, default=0.0, help="stop (and still export) after this long")
    ap.add_argument("--seed", type=int, default=42, help="42 = SFT seed, so --ecg_subset_part 1 is its unseen half")
    return ap.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
