#!/usr/bin/env bash
# Runs one phase on the EC2 instance, mirroring runs/ to S3 every 5 minutes.
# Usage: ec2_run.sh <bucket> <phase> [hard_cap_minutes] [extra train_sp.py args...]
#   phase = pilot | main
set -uo pipefail
BUCKET="$1"; PHASE="$2"; CAP="${3:-240}"; shift 3 || shift $#
WORK=/opt/opentslm
PY="$WORK/venv/bin/python"
TRAIN="$WORK/repo/scripts/lfm3/train_sp.py"
RUNS="$WORK/runs"
export HF_HOME="$WORK/hf" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$RUNS"
cd "$WORK/repo"

shutdown -h +"$CAP" "opentslm hard cap" || true

status() { echo "{\"phase\":\"$PHASE\",\"status\":\"$1\",\"time\":\"$(date -u +%FT%TZ)\"}" > "$RUNS/status_$PHASE.json"; aws s3 cp -q "$RUNS/status_$PHASE.json" "s3://$BUCKET/runs/status_$PHASE.json"; }
sync_runs() { aws s3 sync -q "$RUNS" "s3://$BUCKET/runs" --exclude "*.pt" --exclude "*/hf/*"; }
( while true; do sleep 300; sync_runs; done ) &
SYNC_PID=$!

status RUNNING
rc=0
case "$PHASE" in
  pilot)
    # 1) throughput / memory probe on every stage (also downloads all datasets)
    "$PY" "$TRAIN" --out_dir "$RUNS/probe" --max_steps 60 --log_every 20 --skip_test "$@" \
      > "$RUNS/probe.out" 2>&1 || rc=$?
    # 2) Base vs post-trained A/B: Sleep-CoT trained directly, 2 epochs, full test set
    "$PY" "$TRAIN" --llm_id LiquidAI/LFM3-1.7B-Base --out_dir "$RUNS/ab_base" \
      --stages stage4_sleep_cot --epochs stage4_sleep_cot=2 "$@" > "$RUNS/ab_base.out" 2>&1 || rc=$?
    "$PY" "$TRAIN" --llm_id LiquidAI/LFM3-1.7B --chat_template --out_dir "$RUNS/ab_instruct" \
      --stages stage4_sleep_cot --epochs stage4_sleep_cot=2 "$@" > "$RUNS/ab_instruct.out" 2>&1 || rc=$?
    ;;
  main)
    "$PY" "$TRAIN" --out_dir "$RUNS/main" "$@" > "$RUNS/main.out" 2>&1 || rc=$?
    ;;
  *) echo "unknown phase $PHASE"; rc=2 ;;
esac

kill "$SYNC_PID" 2>/dev/null
sync_runs
# keep best checkpoints too
aws s3 sync -q "$RUNS" "s3://$BUCKET/runs" --exclude "*" --include "*/best.pt"
if [ "$rc" -eq 0 ]; then status DONE; else status "FAILED_rc$rc"; fi
