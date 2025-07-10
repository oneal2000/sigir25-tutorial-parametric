echo "Augmenting popqa with llama3.2-1b-instruct"
python src/augment.py \
    --model_name llama3.2-1b-instruct \
    --dataset popqa \
    --data_path data/popqa/ \
    --sample 300  \
    --topk 3

echo "Augmenting popqa with qwen2.5-1.5b-instruct"
python src/augment.py \
    --model_name qwen2.5-1.5b-instruct \
    --dataset popqa \
    --data_path data/popqa/ \
    --sample 300  \
    --topk 3

echo "Augmenting complexwebquestions with llama3.2-1b-instruct"
python src/augment.py \
    --model_name llama3.2-1b-instruct \
    --dataset complexwebquestions \
    --data_path data/complexwebquestions/ \
    --sample 300  \
    --topk 3

echo "Augmenting complexwebquestions with qwen2.5-1.5b-instruct"
python src/augment.py \
    --model_name qwen2.5-1.5b-instruct \
    --dataset complexwebquestions \
    --data_path data/complexwebquestions/ \
    --sample 300  \
    --topk 3
