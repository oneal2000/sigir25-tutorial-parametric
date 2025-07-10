python3 -u src/train_dyprag.py \
    --model_name=llama3.2-1b-instruct \
    --datasets="2wikimultihopqa,popqa" \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32 \
    --max_new_tokens=128 \
    --sample_rate=1 \
    --dyprag_learning_rate=1e-5 \
    --dyprag_train_epochs=1 \