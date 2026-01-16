#!/bin/bash

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=1200

# Cluster config (adjust as needed)
MASTER_ADDR=localhost
MASTER_PORT=6021
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=1
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# Modeling and data paths (update these to your setup)
# CHECKPOINT="/sharedata/qiuzhijie/projects/MindSpeed-LLM/outputs/stage1/v0/seq_len=512/20260101_222958/checkpoints"
CHECKPOINT="/sharedata/qiuzhijie/projects/MindSpeed-LLM/outputs/stage3/v0/seq_len=512/layer-num=8_20260108_004243/checkpoints"
TOKENIZER_PATH="/sharedata/data/models/Qwen3-0.6B-Base"

# Teacher checkpoints: align with pretrain_ours_stage3.sh
TEACHER1_LOAD="/sharedata/qiuzhijie/projects/MindSpeed-LLM/outputs/stage1/v0/seq_len=512/20260101_222958/checkpoints"
TEACHER2_LOAD="/sharedata/qiuzhijie/projects/MindSpeed-LLM/outputs/stage2/v0/seq_len=512/20260107_231311/checkpoints"

# Model structure must match training (TP/PP/layers/hidden/heads/seq_length, etc.)
TP=1
PP=1
SEQ_LEN=512
NUM_LAYERS=8   # set to the trained student layer count
HIDDEN_SIZE=1024
NUM_HEADS=16

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS inference_ours.py \
    --teacher1-load "${TEACHER1_LOAD}" \
    --teacher1-num-layers 8 \
    --teacher2-load "${TEACHER2_LOAD}" \
    --teacher2-num-layers 4 \
    --use-mcore-models \
    --use-cp-send-recv-overlap \
    --use-fused-ring-attention-update \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path "${TOKENIZER_PATH}" \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_HEADS} \
    --swiglu \
    --padded-vocab-size 151936 \
    --position-embedding-type rope \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-flash-attn \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --tokenizer-not-use-fast \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --micro-batch-size 1 \
    --global-batch-size 1 \
    --load "${CHECKPOINT}" \
    --use-kv-cache \
    --exit-on-missing-checkpoint \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --task chat \
    --max-new-tokens 256