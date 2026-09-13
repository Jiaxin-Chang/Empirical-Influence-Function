#!/bin/bash
# start_vllm.sh

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_FLASHINFER_SAMPLER=0

# 启动vLLM OpenAI兼容服务
python -m vllm.entrypoints.openai.api_server \
--model /mnt/md124/jiaxin/models/Qwen3-8B \
--served-model-name qwen3-8b \
--host 0.0.0.0 \
--port 8001 \
--dtype bfloat16 \
--max-model-len 16384 \
--gpu-memory-utilization 0.85 \
--trust-remote-code
