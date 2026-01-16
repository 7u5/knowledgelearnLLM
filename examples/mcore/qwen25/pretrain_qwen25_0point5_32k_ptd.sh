#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=1200
export PYTHONPATH="/sharedata/qiuzhijie/projects/MindSpeed-LLM:$PYTHONPATH"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include:$CONDA_PREFIX/include/python3.10:$CPLUS_INCLUDE_PATH"

NPUS_PER_NODE=2
MASTER_ADDR=localhost
MASTER_PORT=6011
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="./outputs/$RUN_TAG"
mkdir -p "$OUTPUT_DIR"
CKPT_LOAD_DIR="$OUTPUT_DIR/checkpoints"
CKPT_SAVE_DIR="$OUTPUT_DIR/checkpoints"
TENSORBOARD_LOGS_PATH="$OUTPUT_DIR/log"
TOKENIZER_PATH="/sharedata/qiuzhijie/hf_models/Qwen2.5-1.5B-Instruct"

# DATASET_BASE_DIR="/home/megatron_pretrain_datasets"
# DATA_PATH=(
#     "$DATASET_BASE_DIR/ChineseWebText2_0.75_0.056_0" 
#     "$DATASET_BASE_DIR/DCLM-pro_0.6_tokens_0.0167_0" 
#     "$DATASET_BASE_DIR/PreSelect-100B_0.5_sampled_0.0513_0"
# )
DATA_PATH="/sharedata/zimoliu/data/alpaca_zh_text_document"


TP=1
PP=1
CP=1
MBS=8
GBS=1024
SEQ_LEN=1024
CP_ALGO=megatron_cp_algo

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-mcore-models \
    --use-cp-send-recv-overlap \
    --use-fused-ring-attention-update \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ${CP_ALGO} \
    --sequence-parallel \
    --use-distributed-optimizer \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --group-query-attention \
    --num-query-groups 2 \
    --num-layers 8 \
    --hidden-size 1024 \
    --ffn-hidden-size 4096 \
    --num-attention-heads 16 \
    --next-latent-prediction-steps 1 \
    --lambda-next-h 0.5 \
    --lambda-kl 0.1 \
    --nextlp-loss-scaling-factor 1.0 \
    --rotary-base 1000000 \
    --normalization RMSNorm \
    --norm-epsilon 1e-06 \
    --swiglu \
    --add-qkv-bias \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --lr 1.25e-6 \
    --train-iters 2000 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 \
    --init-method-std 0.01 \
    --position-embedding-type rope \
    --use-fused-rmsnorm \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --overlap-grad-reduce \
    --use-flash-attn \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --seed 42 \
    --bf16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

EVAL_AND_LOGGING_ARGS=(
    --log-interval 10 
    --log-throughput 
    --log-timers-to-tensorboard 
    --save-interval 10000 
    --eval-interval 1000 
    --save $CKPT_SAVE_DIR 
    --load $CKPT_LOAD_DIR 
    --eval-iters 10 
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

DEBUG_LOGGING_ARGS=(
    --timing-log-level 2 
    # --no-mmap-bin-files
    # --mock-data 
)

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${DEBUG_LOGGING_ARGS[@]} \
    --distributed-backend nccl \
    | tee $OUTPUT_DIR/train_mcore_qwen25_0point5b_32k.log
