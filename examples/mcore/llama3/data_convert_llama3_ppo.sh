# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./dataset/llama3-hf/

python ./preprocess_data.py \
    --input ./dataset/descriptiveness-00000-of-00001.parquet \
    --tokenizer-name-or-path ./model_from_hf/llama3-hf/ \
    --output-prefix ./dataset/llama3-hf/descriptiveness \
    --workers 16 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --handler-name  PPOAlpacaStyleInstructionHandler \
    --prompt-type llama3 \
    --map-keys '{"prompt":"prompt", "query":"", "response": "prompt", "system":""}'