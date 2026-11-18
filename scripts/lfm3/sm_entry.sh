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
    # Multi-GPU: train (+export) with torchrun, vLLM test generation (one shard per GPU), merge + score.
    # Env: TRAIN_ARGS (train_sp.py), EVAL_ARGS (vllm_eval.py, incl. --stage), HF_ARGS (HF fallback/compare),
    #      HF_COMPARE=1 to also run HF generation for an agreement check, SKIP_TRAIN=1 to reuse $OUT/main.
    NGPU=$(nvidia-smi -L | wc -l)
    STAGE=$(echo "${EVAL_ARGS:-}" | grep -oE 'stage[0-9]_[a-z_]+' | head -1); STAGE=${STAGE:-stage5_ecg_cot}
    if [ "${SKIP_TRAIN:-0}" != 1 ]; then
      torchrun --standalone --nproc_per_node "$NGPU" "$TRAIN" --out_dir "$OUT/main" --export_dir "$OUT/export" \
        --skip_test ${TRAIN_ARGS:-} 2>&1 | tee "$OUT/main.out" | grep --line-buffered -E '^\{|Error|Traceback'
      rc=${PIPESTATUS[0]}
      [ "$rc" -ne 0 ] && { echo "PHASE_DONE rc=$rc (training failed)"; exit $rc; }
    fi
    # vLLM in its own environment (it pins its own torch/transformers)
    VENV=$WORK/vllm_env; vrc=0
    pip install -q uv && uv venv -q --python 3.12 "$VENV" && \
      VIRTUAL_ENV=$VENV uv pip install -q vllm chronos-forecasting pandas wfdb datasets scikit-learn tqdm einops \
        matplotlib requests huggingface-hub && VIRTUAL_ENV=$VENV uv pip install -q -e "$REPO" --no-deps || vrc=1
    "$VENV/bin/python" -c "import vllm, torch, transformers; print('vllm', vllm.__version__, torch.__version__, transformers.__version__)" || vrc=1
    mkdir -p "$OUT/vllm"
    if [ "$vrc" -eq 0 ]; then
      for i in $(seq 0 $((NGPU - 1))); do
        CUDA_VISIBLE_DEVICES=$i "$VENV/bin/python" "$REPO/scripts/lfm3/vllm_eval.py" --export_dir "$OUT/export" \
          --out "$OUT/vllm/test_predictions.jsonl" --shard "$i" --num_shards "$NGPU" ${EVAL_ARGS:-} \
          > "$OUT/vllm/shard$i.log" 2>&1 &
      done
      wait
      for i in $(seq 0 $((NGPU - 1))); do tail -n 2 "$OUT/vllm/shard$i.log"; grep -q vllm_shard_done "$OUT/vllm/shard$i.log" || vrc=1; done
    fi
    if [ "${HF_COMPARE:-0}" = 1 ] || [ "$vrc" -ne 0 ]; then
      echo "running HF generation (compare=${HF_COMPARE:-0}, vllm_rc=$vrc)"
      torchrun --standalone --nproc_per_node "$NGPU" "$TRAIN" --out_dir "$OUT/hf" --stages "$STAGE" --eval_only \
        --init_from "$OUT/main/$STAGE/best.pt" ${HF_ARGS:-} 2>&1 | tee "$OUT/hf.out" | grep --line-buffered -E '^\{|Error|Traceback'
      rc=${PIPESTATUS[0]}
    fi
    if [ "$vrc" -eq 0 ]; then
      CMP=""; [ -f "$OUT/hf/$STAGE/test_predictions.jsonl" ] && CMP="--compare $OUT/hf/$STAGE/test_predictions.jsonl"
      python "$REPO/scripts/lfm3/vllm_eval.py" --merge --stage "$STAGE" --out "$OUT/vllm/test_predictions.jsonl" \
        --num_shards "$NGPU" $CMP 2>&1 | grep -E '^\{|Error|Traceback'; rc=$((rc | ${PIPESTATUS[0]}))
    fi
    ;;
  *) echo "unknown PHASE=${PHASE:-}"; rc=2 ;;
esac
echo "PHASE_DONE rc=$rc"
exit $rc
