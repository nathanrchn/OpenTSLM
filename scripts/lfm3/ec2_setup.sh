#!/usr/bin/env bash
# One-time setup on the EC2 GPU instance (Ubuntu 24.04 DLAMI base). Idempotent.
set -euo pipefail
WORK=/opt/opentslm
BUCKET="$1"
mkdir -p "$WORK" && cd "$WORK"

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
fi

aws s3 cp "s3://$BUCKET/code/repo.tar.gz" /tmp/repo.tar.gz
mkdir -p repo && tar -xzf /tmp/repo.tar.gz -C repo

if [ ! -x venv/bin/python ]; then
  uv venv -q --python 3.12 venv
  VIRTUAL_ENV=$WORK/venv uv pip install -q torch==2.15.0 "transformers==5.23.0" "peft==0.21.3" chronos-forecasting \
    datasets pandas numpy scikit-learn wfdb tqdm einops requests matplotlib huggingface-hub
  # fall back to a CUDA 12.8 build if the default wheel does not match the driver
  if ! "$WORK/venv/bin/python" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    VIRTUAL_ENV=$WORK/venv uv pip install -q --reinstall torch==2.15.0 --index-url https://download.pytorch.org/whl/cu128
  fi
fi
VIRTUAL_ENV=$WORK/venv uv pip install -q -e repo --no-deps

# Idle watchdog: stop the instance if no training process has run for 3 checks (15 min).
cat >/usr/local/bin/opentslm-watchdog.sh <<'EOF'
#!/usr/bin/env bash
idle=0
while true; do
  sleep 300
  if pgrep -f "train_sp.py|ec2_run.sh|ec2_setup.sh" >/dev/null; then idle=0; else idle=$((idle+1)); fi
  if [ "$idle" -ge 3 ]; then logger "opentslm watchdog: idle, shutting down"; shutdown -h now; fi
done
EOF
chmod +x /usr/local/bin/opentslm-watchdog.sh
cat >/etc/systemd/system/opentslm-watchdog.service <<'EOF'
[Unit]
Description=Stop instance when OpenTSLM training is idle
[Service]
ExecStart=/usr/local/bin/opentslm-watchdog.sh
Restart=always
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now opentslm-watchdog.service

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$WORK/venv/bin/python" -c "import torch, transformers, peft; print(torch.__version__, torch.cuda.is_available(), transformers.__version__, peft.__version__)"
echo SETUP_OK
