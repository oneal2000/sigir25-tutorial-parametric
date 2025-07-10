python3 -u src/train_dyprag.py \
    --model_name=qwen2.5-1.5b-instruct \
    --datasets="2wikimultihopqa,popqa" \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32 \
    --max_new_tokens=128 \
    --sample_rate=1 \
    --dyprag_learning_rate=1e-5 \
    --dyprag_train_epochs=1 \