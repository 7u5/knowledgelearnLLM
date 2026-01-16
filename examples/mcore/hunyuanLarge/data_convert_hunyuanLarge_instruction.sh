# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 可使用hunyuanLarge的github开源仓数据
mkdir -p ./finetune_dataset
python ./preprocess_data.py \
    --input ./dataset/Hunyuan/car_train.jsonl \
    --tokenizer-name-or-path ./Tencent-Hunyuan-Large/Hunyuan-A52B-Instruct \
    --output-prefix ./finetune_dataset/hunyuan \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --handler-name HunyuanInstructionHandler \
    --prompt-type hunyuan \
    --overwrite-cache \
    --map-keys '{"messages":"messages", "tags":{"role_tag": "role","content_tag": "content","user_tag": "user","assistant_tag": "assistant","system_tag": "system"}}'
