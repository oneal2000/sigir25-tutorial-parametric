python3 src/inference.py \
    --model_name=llama3.2-1b-instruct \
    --dataset=popqa \
    --sample=300 \
    --num_train_epochs=2 \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32 \
    --max_new_tokens=20 \
    --inference_method=prag 