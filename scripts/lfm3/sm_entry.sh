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
  *) echo "unknown PHASE=${PHASE:-}"; rc=2 ;;
esac
echo "PHASE_DONE rc=$rc"
exit $rc
