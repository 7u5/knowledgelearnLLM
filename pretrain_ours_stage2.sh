#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=1200
# export PYTHONPATH="/sharedata/qiuzhijie/projects/MindSpeed-LLM:$PYTHONPATH"
# export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include:$CONDA_PREFIX/include/python3.10:$CPLUS_INCLUDE_PATH"
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
MBS=64
GBS=1024
SEQ_LEN=512
CP_ALGO=megatron_cp_algo

LOSS_TYPE=$1  # info_nce or l2
LR=$2
WARM_UP=0.01

if [ "$LOSS_TYPE" == "l2" ]; then
    VERSION_TAG="v0"
elif [ "$LOSS_TYPE" == "info_nce" ]; then
    VERSION_TAG="v01"
    PROJ_DIM=256
    PROJ_LAYERS=2
    TEMP=0.07
else 
    echo "Unsupported loss type: ${LOSS_TYPE}. Supported types are 'l2' and 'info_nce'."
    exit 1
fi

DATETIME="$(date +%Y%m%d_%H%M%S)"
if [ "$VERSION_TAG" == "v0" ]; then
    RUN_TAG="${DATETIME}"
elif [ "$VERSION_TAG" == "v01" ]; then
    RUN_TAG="${LOSS_TYPE}_pd${PROJ_DIM}_pl${PROJ_LAYERS}_t${TEMP}_${DATETIME}"
fi
OUTPUT_DIR="./outputs/stage2/${VERSION_TAG}/seq_len=${SEQ_LEN}/$RUN_TAG"
mkdir -p "$OUTPUT_DIR"
CKPT_LOAD_DIR="$OUTPUT_DIR/checkpoints"
CKPT_SAVE_DIR="$OUTPUT_DIR/checkpoints"
LOGS_PATH="$OUTPUT_DIR/log"
TOKENIZER_PATH="/sharedata/data/models/Qwen3-0.6B-Base"

# DATASET_BASE_DIR="/sharedata/data/indexed_data/Nemotron-CC-v2/High-Quality"
# DATA_PATH=""
# for i in {0..20}; do
#     PART_NAME=$(printf "part_%06d_text_document" $i)
#     DATA_PATH="$DATA_PATH $DATASET_BASE_DIR/$PART_NAME"
# done
# DATA_PATH="/sharedata/qiuzhijie/datasets/pretrain_0point3_stage1"
DATASET_BASE_DIR="/sharedata/data/indexed_data/dclm-baseline-1.0/global-shard_01_of_10/local-shard_1_of_10"
DATA_PATH=""
for i in {0..60}; do
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

STAGE2_ARGS=(
    --teacher-load "/sharedata/qiuzhijie/projects/MindSpeed-LLM/outputs/stage1/v0/seq_len=512/20260101_222958/checkpoints"
    --teacher-num-layers 8
    --hidden-loss-type "${LOSS_TYPE}"
)

if [ "$VERSION_TAG" == "v01" ]; then
    STAGE2_ARGS+=(
        --hidden-proj-dim "${PROJ_DIM}"
        --hidden-proj-layers "${PROJ_LAYERS}"
        --hidden-proj-temperature "${TEMP}"
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
    --num-layers 4 \
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
    --no-load-optim \
    --no-load-rng \
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
    --lr ${LR}
    --lr-decay-style cosine
    --min-lr 1.0e-7
    --lr-warmup-fraction ${WARM_UP}
    --init-method-std 0.006
    --clip-grad 1.0
    --use-distributed-optimizer
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

torchrun $DISTRIBUTED_ARGS pretrain_gpt_stage2.py \
    $GPT_ARGS \
    ${STAGE2_ARGS[@]} \
    ${TRAIN_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    $OUTPUT_ARGS \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${DEBUG_LOGGING_ARGS[@]} \
    ${RERUN_ARGS[@]} \
    --distributed-backend nccl \
    | tee $OUTPUT_DIR/train_mcore_qwen25_0point5b_32k.log
