#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=1200
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
# export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
# export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11
# export ASCEND_RT_VISIBLE_DEVICES=12,13,14,15
# NPUS_PER_NODE=4

# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6011
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=1
PP=1
CP=1
MBS=16
GBS=1024
SEQ_LEN=512
CP_ALGO=megatron_cp_algo

VERSION=$1
MAX_LR=$2
LAYER_NUM=$3

if [ "$VERSION" != "v0" ] && [ "$VERSION" != "v01" ]; then
    echo "Version (v0 or v01) argument is required."
    exit 1
fi

if [ -z "$MAX_LR" ]; then
    echo "Max learning rate argument is required."
    exit 1
fi

if [ -z "$LAYER_NUM" ]; then
    echo "Layer number argument is required."
    exit 1
fi

# please fill these path configurations
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="./outputs/stage3/${VERSION}/seq_len=${SEQ_LEN}/layer-num=${LAYER_NUM}_$RUN_TAG"
mkdir -p "$OUTPUT_DIR"
CKPT_LOAD_DIR="$OUTPUT_DIR/checkpoints"
CKPT_SAVE_DIR="$CKPT_LOAD_DIR"
LOGS_PATH="$OUTPUT_DIR/log"
TOKENIZER_PATH="/sharedata/data/models/Qwen3-0.6B-Base"

# DATASET_BASE_DIR="/sharedata/data/indexed_data/Nemotron-CC-v2/High-Quality"
# DATA_PATH=""
# for i in {0..20}; do
#     PART_NAME=$(printf "part_%06d_text_document" $i)
#     DATA_PATH="$DATA_PATH $DATASET_BASE_DIR/$PART_NAME"
# done
# DATA_PATH="/sharedata/qiuzhijie/datasets/pretrain_0point3_stage1"
DATASET_BASE_DIR="/sharedata/data/indexed_data/dclm-baseline-1.0/global-shard_01_of_10/local-shard_2_of_10"
DATA_PATH=""
for i in {0..120}; do
    PART_NAME=$(printf "shard_%08d_processed_text_document" $i)
    DATA_PATH="$DATA_PATH $DATASET_BASE_DIR/$PART_NAME"
done

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

STAGE3_ARGS=(
    --teacher1-load "/sharedata/qiuzhijie/projects/MindSpeed-LLM/outputs/stage1/v0/seq_len=512/20260101_222958/checkpoints"
    --teacher1-num-layers 8
    # --use-teacher-embedding-input
)

if [ "$VERSION" == "v0" ]; then
    STAGE3_ARGS+=(
        --teacher2-load "/sharedata/qiuzhijie/projects/MindSpeed-LLM/outputs/stage2/v0/seq_len=512/20260107_231311/checkpoints"
        --teacher2-num-layers 4
    )
elif [ "$VERSION" == "v01" ]; then
    exit 1 # error out to avoid misuse
    STAGE3_ARGS+=(
        --teacher2-load "/sharedata/qiuzhijie/projects/MindSpeed-LLM/outputs/stage2/v01/seq_len=512/info_nce_pd256_pl2_t0.07_20251222_191307/checkpoints"
        --teacher2-num-layers 4
    )
fi

GPT_ARGS="
    --use-mcore-models \
    --use-cp-send-recv-overlap \
    --use-fused-ring-attention-update \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --num-layers ${LAYER_NUM} \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --swiglu \
    --padded-vocab-size 151936 \
    --position-embedding-type rope \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --overlap-grad-reduce \
    --use-flash-attn \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --bf16 \
    --seed 42
"

TRAIN_ARGS=(
    --micro-batch-size ${MBS}
    --global-batch-size ${GBS}
    --train-iters 10000
    --weight-decay 1e-1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --lr ${MAX_LR}
    --lr-decay-style cosine
    --min-lr 1.0e-6
    --lr-warmup-fraction 0.01
    --init-method-std 0.006
    --clip-grad 1.0
    --use-distributed-optimizer
    --override-opt_param-scheduler
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --context-parallel-size ${CP}
    --context-parallel-algo ${CP_ALGO}
    --sequence-parallel
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --split 949,50,1
    --num-workers 8
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 10 
    --log-throughput 
    --log-timers-to-tensorboard 
    --save-interval 1000 
    --eval-interval 1000 
    --save $CKPT_SAVE_DIR 
    --load $CKPT_LOAD_DIR 
    --eval-iters 10 
    --tensorboard-dir $LOGS_PATH 
)

DEBUG_LOGGING_ARGS=(
    --timing-log-level 2 
    # --no-mmap-bin-files
    # --mock-data 
)

# Enable rerun engine to validate results and retry on NaNs/Infs.
RERUN_ARGS=(
    --rerun-mode validate_results
    # Optional diagnostics (non-fatal):
    # --check-for-large-grads
    # --check-for-spiky-loss
)

torchrun $DISTRIBUTED_ARGS pretrain_gpt_stage3.py \
    $GPT_ARGS \
    ${STAGE3_ARGS[@]} \
    ${TRAIN_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    $OUTPUT_ARGS \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${DEBUG_LOGGING_ARGS[@]} \
    ${RERUN_ARGS[@]} \
    --distributed-backend nccl \
    | tee $OUTPUT_DIR/train_mcore_qwen25_0point5b_32k.log
