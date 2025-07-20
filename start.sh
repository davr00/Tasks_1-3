#!/bin/bash
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate task_1-3

vllm serve \
  --port 8080 \
  Qwen/Qwen3-8B \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --max-model-len 64000 \
  --swap-space 8 \
  --gpu-memory-utilization 0.7 &

uvicorn main:app --host 0.0.0.0 --port 3000 --reload &

wait
