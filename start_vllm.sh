#!/bin/bash
# start_vllm.sh

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_FLASHINFER_SAMPLER=0

# 启动vLLM OpenAI兼容服务
# 32B bf16 权重约 64GB：单卡 80GB 可起；多卡时取消下一行注释并改 CUDA_VISIBLE_DEVICES
# export CUDA_VISIBLE_DEVICES=0,1
python -m vllm.entrypoints.openai.api_server \
--model /mnt/md124/jiaxin/models/Qwen2.5-Coder-32B-Instruct \
--served-model-name qwen2.5-coder-32b-instruct \
--host 0.0.0.0 \
--port 8001 \
--dtype bfloat16 \
--max-model-len 8192 \
--gpu-memory-utilization 0.90 \
--trust-remote-code
# 多卡示例：再加  --tensor-parallel-size 2
