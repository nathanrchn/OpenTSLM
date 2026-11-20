#!/usr/bin/env bash
# SageMaker training-job entrypoint (PyTorch DLC). Env: PHASE=pilot|main, EXTRA_ARGS="..."
# Everything under /opt/ml/checkpoints is mirrored to S3 by SageMaker while the job runs.
set -uo pipefail
WORK=/opt/ml/work
REPO=$WORK/repo
OUT=/opt/ml/checkpoints/runs
export HF_HOME=$WORK/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"
EXTRA_ARGS=${EXTRA_ARGS:-}

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
free -g | head -2
TV=$(python -c "import torch; print(torch.__version__.split('+')[0])")
echo "torch==$TV" > /tmp/constraints.txt
pip install -q -c /tmp/constraints.txt "transformers==5.23.0" "peft==0.21.3" chronos-forecasting \
  datasets pandas numpy scikit-learn wfdb tqdm einops requests matplotlib huggingface-hub \
  || { echo "PIP_FAILED"; exit 3; }
pip install -q -e "$REPO" --no-deps || { echo "PIP_REPO_FAILED"; exit 3; }
python -c "import torch, transformers, peft; print('versions', torch.__version__, torch.version.cuda, transformers.__version__, peft.__version__, torch.cuda.is_available())"

cd "$REPO"
TRAIN="$REPO/scripts/lfm3/train_sp.py"
rc=0
case "${PHASE:-}" in
  pilot)
    # 1) throughput / memory probe on every stage (also downloads all datasets)
    python "$TRAIN" --out_dir "$OUT/probe" --max_steps 60 --log_every 20 --skip_test $EXTRA_ARGS \
      2>&1 | tee "$OUT/probe.out" | grep --line-buffered -E '^\{|Error|error|Traceback' ; rc=$((rc | ${PIPESTATUS[0]}))
    # 2) Base vs post-trained A/B: Sleep-CoT trained directly, 2 epochs, full test set
    python "$TRAIN" --llm_id LiquidAI/LFM3-1.7B-Base --out_dir "$OUT/ab_base" \
      --stages stage4_sleep_cot --epochs stage4_sleep_cot=2 $EXTRA_ARGS \
      2>&1 | tee "$OUT/ab_base.out" | grep --line-buffered -E '^\{|Error|error|Traceback' ; rc=$((rc | ${PIPESTATUS[0]}))
    python "$TRAIN" --llm_id LiquidAI/LFM3-1.7B --chat_template --out_dir "$OUT/ab_instruct" \
      --stages stage4_sleep_cot --epochs stage4_sleep_cot=2 $EXTRA_ARGS \
      2>&1 | tee "$OUT/ab_instruct.out" | grep --line-buffered -E '^\{|Error|error|Traceback' ; rc=$((rc | ${PIPESTATUS[0]}))
    ;;
  main)
    python "$TRAIN" --out_dir "$OUT/main" $EXTRA_ARGS \
      2>&1 | tee "$OUT/main.out" | grep --line-buffered -E '^\{|Error|error|Traceback' ; rc=${PIPESTATUS[0]}
    ;;
  pipeline)
    # Multi-GPU: train (+export, + test-set dump, + optional HF subset) with torchrun, then vLLM test generation
    # (one shard per GPU), merge + score. Env: TRAIN_ARGS (train_sp.py; add --hf_compare N for an HF/vLLM
    # agreement check), EVAL_ARGS (vllm_eval.py, incl. --stage), HF_ARGS (model flags for the HF fallback),
    # HF_COMPARE=1 forces a separate HF test run, SKIP_TRAIN=1 reuses $OUT/main.
    NGPU=$(nvidia-smi -L | wc -l)
    STAGE=$(echo "${EVAL_ARGS:-}" | grep -oE 'stage[0-9]_[a-z_]+' | head -1); STAGE=${STAGE:-stage5_ecg_cot}
    # vLLM in its own environment (it pins its own torch/transformers); installed while training runs
    VENV=$WORK/vllm_env; vrc=0
    ( pip install -q uv && uv venv -q --python 3.12 "$VENV" && \
      VIRTUAL_ENV=$VENV uv pip install -q vllm chronos-forecasting pandas wfdb datasets scikit-learn tqdm einops \
        matplotlib requests huggingface-hub && VIRTUAL_ENV=$VENV uv pip install -q -e "$REPO" --no-deps && \
      "$VENV/bin/python" -c "import vllm, torch, transformers; print('vllm', vllm.__version__, torch.__version__, transformers.__version__)" \
    ) > "$WORK/vllm_install.log" 2>&1 &
    VPID=$!
    if [ "${SKIP_TRAIN:-0}" != 1 ]; then
      torchrun --standalone --nproc_per_node "$NGPU" "$TRAIN" --out_dir "$OUT/main" --export_dir "$OUT/export" \
        --dump_test "$WORK/test_cache" --skip_test ${TRAIN_ARGS:-} 2>&1 | tee "$OUT/main.out" \
        | grep --line-buffered -E '^\{|Error|Traceback'
      rc=${PIPESTATUS[0]}
      [ "$rc" -ne 0 ] && { echo "PHASE_DONE rc=$rc (training failed)"; exit $rc; }
    fi
    wait $VPID || vrc=1
    tail -n 3 "$WORK/vllm_install.log"; cp "$WORK/vllm_install.log" "$OUT/" 2>/dev/null
    mkdir -p "$OUT/vllm"
    if [ "$vrc" -eq 0 ]; then
      for i in $(seq 0 $((NGPU - 1))); do
        CUDA_VISIBLE_DEVICES=$i "$VENV/bin/python" "$REPO/scripts/lfm3/vllm_eval.py" --export_dir "$OUT/export" \
          --out "$OUT/vllm/test_predictions.jsonl" --shard "$i" --num_shards "$NGPU" --test_cache "$WORK/test_cache" \
          ${EVAL_ARGS:-} > "$OUT/vllm/shard$i.log" 2>&1 &
      done
      wait
      for i in $(seq 0 $((NGPU - 1))); do
        grep -E '^\{' "$OUT/vllm/shard$i.log" | tail -n 1; grep -q vllm_shard_done "$OUT/vllm/shard$i.log" || vrc=1
      done
    fi
    if [ "${HF_COMPARE:-0}" = 1 ] || [ "$vrc" -ne 0 ]; then
      echo "running HF generation (compare=${HF_COMPARE:-0}, vllm_rc=$vrc)"
      torchrun --standalone --nproc_per_node "$NGPU" "$TRAIN" --out_dir "$OUT/hf" --stages "$STAGE" --eval_only \
        --init_from "$OUT/main/$STAGE/best.pt" ${HF_ARGS:-} 2>&1 | tee "$OUT/hf.out" | grep --line-buffered -E '^\{|Error|Traceback'
      rc=${PIPESTATUS[0]}
    fi
    if [ "$vrc" -eq 0 ]; then
      CMP=""
      for f in "$OUT/hf/$STAGE/test_predictions.jsonl" "$OUT/main/$STAGE/test_predictions_hf.jsonl"; do
        [ -f "$f" ] && { CMP="--compare $f"; break; }
      done
      python "$REPO/scripts/lfm3/vllm_eval.py" --merge --stage "$STAGE" --out "$OUT/vllm/test_predictions.jsonl" \
        --num_shards "$NGPU" $CMP 2>&1 | grep -E '^\{|Error|Traceback'; rc=$((rc | ${PIPESTATUS[0]}))
    fi
    ;;
  sample)
    # vLLM only, on an exported model (input channel "export"): e.g. k sampled rationales per question for a
    # headroom check. Env: EVAL_ARGS (vllm_eval.py: --stage/--split/--num_samples/...), OUT_NAME (default: sample).
    NGPU=$(nvidia-smi -L | wc -l)
    EXPORT=/opt/ml/input/data/export
    STAGE=$(echo "${EVAL_ARGS:-}" | grep -oE 'stage[0-9]_[a-z_]+' | head -1); STAGE=${STAGE:-stage5_ecg_cot}
    DEST="$OUT/${OUT_NAME:-sample}"; mkdir -p "$DEST"
    if [ -n "${EXPORT_S3:-}" ]; then  # export in another region's bucket (SageMaker channels must be same-region)
      python - "$EXPORT_S3" "$WORK/export" <<'PY' || { echo "EXPORT_DOWNLOAD_FAILED"; exit 6; }
import os, sys, boto3
uri, dest = sys.argv[1], sys.argv[2]
bucket, prefix = uri[len("s3://"):].split("/", 1)
region = boto3.client("s3").get_bucket_location(Bucket=bucket)["LocationConstraint"] or "us-east-1"
s3 = boto3.client("s3", region_name=region)
n = 0
for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
    for o in page.get("Contents", []):
        path = os.path.join(dest, o["Key"][len(prefix):].lstrip("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        s3.download_file(bucket, o["Key"], path)
        n += 1
print("export_downloaded", n, "files")
PY
      EXPORT=$WORK/export
    fi
    VENV=$WORK/vllm_env
    ( pip install -q uv && uv venv -q --python 3.12 "$VENV" && \
      VIRTUAL_ENV=$VENV uv pip install -q vllm chronos-forecasting pandas wfdb datasets scikit-learn tqdm einops \
        matplotlib requests huggingface-hub && VIRTUAL_ENV=$VENV uv pip install -q -e "$REPO" --no-deps ) \
      > "$DEST/vllm_install.log" 2>&1 &
    VPID=$!
    if [ "$STAGE" = stage5_ecg_cot ]; then  # one download before the per-GPU processes start
      python -c "import sys; sys.path.insert(0, '$REPO/scripts/lfm3'); import train_sp; train_sp.ensure_ecg_downloaded()" \
        > "$DEST/download.log" 2>&1 || { echo "DOWNLOAD_FAILED"; tail -n 20 "$DEST/download.log"; exit 4; }
    fi
    wait $VPID || { echo "VLLM_INSTALL_FAILED"; tail -n 20 "$DEST/vllm_install.log"; exit 5; }
    for i in $(seq 0 $((NGPU - 1))); do
      CUDA_VISIBLE_DEVICES=$i "$VENV/bin/python" "$REPO/scripts/lfm3/vllm_eval.py" --export_dir "$EXPORT" \
        --out "$DEST/predictions.jsonl" --shard "$i" --num_shards "$NGPU" ${EVAL_ARGS:-} > "$DEST/shard$i.log" 2>&1 &
    done
    wait
    for i in $(seq 0 $((NGPU - 1))); do
      grep -E '^\{' "$DEST/shard$i.log" | tail -n 1; grep -q vllm_shard_done "$DEST/shard$i.log" || rc=1
    done
    [ "$rc" -ne 0 ] && { tail -n 30 "$DEST/shard0.log"; echo "PHASE_DONE rc=$rc (generation failed)"; exit $rc; }
    python "$REPO/scripts/lfm3/vllm_eval.py" --merge --out "$DEST/predictions.jsonl" --num_shards "$NGPU" \
      ${EVAL_ARGS:-} 2>&1 | grep -E '^\{|Error|Traceback'; rc=${PIPESTATUS[0]}
    ;;
  *) echo "unknown PHASE=${PHASE:-}"; rc=2 ;;
esac
echo "PHASE_DONE rc=$rc"
exit $rc
