echo "Encoding popqa with llama3.2-1b-instruct"
python3 src/encode.py \
    --model_name=llama3.2-1b-instruct \
    --dataset=popqa \
    --sample=300 \
    --per_device_train_batch_size=1 \
    --num_train_epochs=2 \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32 

echo "Encoding popqa with qwen2.5-1.5b-instruct"
python3 src/encode.py \
    --model_name=qwen2.5-1.5b-instruct \
    --dataset=popqa \
    --sample=300 \
    --per_device_train_batch_size=1 \
    --num_train_epochs=2 \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32 

echo "Encoding complexwebquestions with llama3.2-1b-instruct"
python3 src/encode.py \
    --model_name=llama3.2-1b-instruct \
    --dataset=complexwebquestions \
    --sample=300 \
    --per_device_train_batch_size=1 \
    --num_train_epochs=1 \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32 

echo "Encoding complexwebquestions with qwen2.5-1.5b-instruct"
python3 src/encode.py \
    --model_name=qwen2.5-1.5b-instruct \
    --dataset=complexwebquestions \
    --sample=300 \
    --per_device_train_batch_size=1 \
    --num_train_epochs=1 \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32 