#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6012
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=1
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CHECKPOINT="/sharedata/qiuzhijie/projects/MindSpeed-LLM/outputs/moe/seq_len=512/naive/warmup=0.03_20260103_130259/checkpoints"
# CHECKPOINT="/sharedata/qiuzhijie/projects/MindSpeed-LLM/outputs/moe/seq_len=512/naive/warmup=0.03_muon_20260108_202248/checkpoints"
TOKENIZER_PATH="/sharedata/data/models/Qwen3-0.6B-Base"
DATA_PATH="/sharedata/qiuzhijie/projects/MindSpeed-LLM/eval_dataset/boolq"
TASK="boolq"

TP=1
PP=1
MBS=1
SEQ_LEN=512
NUM_LAYERS=8   # set to the trained student layer count
HIDDEN_SIZE=1024
NUM_HEADS=16

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS=(
    --moe-token-dispatcher-type alltoall_seq
    --moe-permute-fusion
    --first-k-dense-replace 1
    --moe-layer-freq 1
    --n-shared-experts 1
    --num-experts 8
    --moe-router-topk 2
    --moe-ffn-hidden-size 768
    --moe-router-load-balancing-type pai_megatron_aux_loss
    --moe-aux-loss-coeff 0.01
    --seq-aux
    --disable-bias-linear
)

# Different task needs different max_new_tokens value, please follow the instruction in readme.
torchrun $DISTRIBUTED_ARGS evaluation.py \
        ${MOE_ARGS[@]} \
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
        --task ${TASK} \
        --task-data-path ${DATA_PATH} \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --max-new-tokens 1 \
        --micro-batch-size ${MBS} \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --no-chat-template
