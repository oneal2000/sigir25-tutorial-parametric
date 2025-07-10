python3 src/encode.py \
    --model_name=llama3-8b-instruct \
    --dataset=complexwebquestions \
    --sample=300 \
    --per_device_train_batch_size=1 \
    --num_train_epochs=1 \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32 