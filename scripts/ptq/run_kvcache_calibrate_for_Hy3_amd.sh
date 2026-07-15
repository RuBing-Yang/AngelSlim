#!/bin/bash

export PYTHONPATH=.


# Allow function serialization for apply_model in vLLM v1 engine
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
# Force Ray to reload code (disable code caching)
export RAY_DEDUP_LOGS=0
# Force Python to not use bytecode cache
export PYTHONDONTWRITEBYTECODE=1
# NOTE: VLLM_MOE_COLLECT_STATS is intentionally NOT set here –
#       this script only calibrates kv-cache, not weight/activation/MoE.

export VLLM_ENABLE_CHUNKED_PREFILL=1
export VLLM_DISTRIBUTED_EXECUTOR_BACKEND=mp
export MOE_MODE=fused
# export VLLM_ATTENTION_BACKEND=FLASHINFER
export ASYNC_SCHEDULING=1
export VLLM_ENABLE_PREFIX_CACHING=1
export PRECISIONMODE=HF


# For AMD
export AITER_ENABLE_VSKIP=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export AITER_LOG_MORE=0
export ASYNC_SCHEDULING=1
export NUM_MTP_TOKENS=1
export MTP_METHOD=hunyuan_v3_mtp
export VLLM_TOOL_PARSER=hunyuan_v21
export REASONING_PARSER=hunyuan_v3
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
# export VLLM_ENABLE_PREFIX_CACHING=0
export AITER_BYPASS_TUNE_CONFIG=0
export AITER_ONLINE_TUNE=0
export BLOCK_SIZE=16
export VLLM_MAX_NUM_BATCHED_TOKENS=32768
export MAX_INPUT_TOKEN_LEN=196608
export EXTRA_PARAMS=--attention_backend "ROCM_MHA_ATTN"
export VLLM_ALLREDUCE_FUSION=0
export VLLM_ROPE_NORM_KVCACHE_FUSE=1
export VLLM_USE_HCF_SPEC_SAMPLER=1
export VLLM_HCF_SPEC_ACCEPT_STRATEGY=strict
export VLLM_ENGINE_READY_TIMEOUT_S=2700
export VLLM_MAX_WAIT_REQS=16
export VLLM_TRUNCATE_PROMPT_TYPE=none
export VLLM_SUSPEND_WHEN_PREEMPTED=1
export USE_REASONING=True
export VLLM_TOOL_KEEPALIVE_INTERVAL=10
export VLLM_USE_HCF_SAMPLER=1
export VLLM_ROCM_MHA_BATCH_PREFILL_USE_TORCH_REF=1
export HSA_TOOLS_LIB=/opt/rocm/lib/librocm-debug-agent.so.2
export ROCM_DEBUG_AGENT_OPTIONS="--all"
export HSA_ENABLE_DEBUG=1

export VLLM_USE_V1=1
export VLLM_RPC_TIMEOUT=1800000
export VLLM_TRUNCATE_PROMPT_TOKENS=196608
export VLLM_DISABLE_PD_CROSSLAYER_CACHE=1


CONFIG=configs/Hy3/ptq/fp8/Hy3_kvcache_calibrate.yaml

mkdir -p logs

python3 tools/kvcache/run_kvcache_calibrate.py \
    -c $CONFIG \
    2>&1 | tee logs/run_kvcache_calibrate_Hy3.log

