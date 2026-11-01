#!/bin/bash
# Zero-shot labeling with Qwen3.8-27B (Qwen3.5-family hybrid VL model). Set PYTHON to an interpreter with vLLM >= 0.17.
cd "$(dirname "$0")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} VLLM_WORKER_MULTIPROC_METHOD=spawn
export GPU_UTIL=${GPU_UTIL:-0.85} MAX_NUM_SEQS=${MAX_NUM_SEQS:-16} ENFORCE_EAGER=${ENFORCE_EAGER:-1}
export LLM_EXTRA_JSON='{"limit_mm_per_prompt": {"image": 0, "video": 0}}'
echo "start $(date)"
${PYTHON:-python} llm_label_v2.py Qwen/Qwen3.8-27B qwen38_27b
echo "exit $? $(date)"
