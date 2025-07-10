# Adopted from DyPRAG:https://github.com/Trae1ounG/DyPRAG
import os
import gc
import json
import argparse
import torch
from tqdm import tqdm

import prompt_template
from root_dir_path import ROOT_DIR
from utils import get_model, evaluate, predict, load_data, read_complete, delta_inject, delta_remove
from projector import ParameterTranslator
from peft import LoraConfig, TaskType, get_peft_model

def main(args):
    data_list = load_data(args.dataset, args.data_type, args.augment_model, data_dir=None)
    model, tokenizer, generation_config = get_model(
        args.model_name,
        max_new_tokens = args.max_new_tokens,
    )
    tokenizer.pad_token = tokenizer.eos_token
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=['down_proj', 'gate_proj', 'up_proj'],
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
    )
    
    # Convert to PeftModel
    model = get_peft_model(model, peft_config)
    if args.with_cot:
        prompt_template.get_fewshot(args.dataset)

    projector_path = os.path.join(ROOT_DIR, "projector", args.projector_path, f"epoch_{args.inference_epoch-1}.pt")
    projector = ParameterTranslator(
        ["down_proj", "up_proj", "gate_proj"],
        list(range(model.config.num_hidden_layers)),
        model.config.hidden_size,
        model.config.intermediate_size,
        args.lora_rank,
        args.projector_p
    ).to(model.device)

    projector.load_state_dict(torch.load(projector_path, map_location=model.device)['model_state_dict'])
    projector.eval()
    
    cot_name = "cot" if args.with_cot else "direct"
    output_root_dir = os.path.join(
        ROOT_DIR, 
        "output",
        "DyPRAG",
        args.model_name, 
        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
        args.dataset,
        f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
        f"aug_model={args.augment_model}",
        f"projector_inference_{args.inference_method}_{args.projector_path}_epoch{args.inference_epoch}", 
    )
    
    for filename, fulldata in data_list:
        filename = filename.split(".")[0]
        print(f"### Solving {filename} ###")
        output_dir = os.path.join(output_root_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.json"), "w") as fout:
            json.dump(vars(args), fout, indent=4)

        predict_file = os.path.join(output_dir, "predict.json")
        ret, start_with = read_complete(predict_file)
        for test_id, data in tqdm(enumerate(fulldata), total=len(fulldata)):
            test_id = test_id + start_with
            assert test_id == len(ret), f"test_id {test_id} != len(ret) {len(ret)}"
            question = data["question"]
            passages = data["passages"]
            answer = data["answer"]
            def get_pred(model, psgs):
                text = predict(model, tokenizer, generation_config, 
                             question, with_cot=args.with_cot, 
                             passages=psgs)
                pred = {
                    "test_id": test_id, 
                    "question": question, 
                    "answer": answer, 
                    "text": text,
                }
                pred.update(evaluate(text, answer, args.with_cot))
                return pred

            if args.inference_method == 'icl':
                ret.append(get_pred(model, psgs=passages))
            else:
                all_deltas = []
                for passage in passages:
                    tokens = tokenizer(
                        passage,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=3000
                    ).to(model.device)
                    with torch.no_grad():
                        output = model(tokens.input_ids, output_hidden_states=True)
                        input_embeds = output.hidden_states[-1][:,-1,:] # last layer, last token
                        outputs = projector(input_embeds)
                        all_deltas.append(outputs)
                merged_deltas = {}
                for key in all_deltas[0].keys():
                    merged_deltas[key] = torch.stack([delta[key] for delta in all_deltas]).mean(dim=0)
                
                # Inject LoRA
                delta_inject(model, merged_deltas)
                ret.append(get_pred(model, psgs=None if args.inference_method == "dyprag" else passages))# 2 ways of inference
                # Remove LoRA
                delta_remove(model, merged_deltas)
                del all_deltas, merged_deltas
                torch.cuda.empty_cache()
                gc.collect()

        with open(predict_file, "w") as fout:
            json.dump(ret, fout, indent=4)
        # ##### Evaluating #####
        metrics = ["em", "f1", "prec", "recall"]
        ret_str = ""
        for met in metrics:
            acc = sum(float(d[met]) for d in ret) / len(ret)
            acc = round(acc, 4)
            ret_str += f"{met}\t{acc}\n"
        ret_str += "\n" + json.dumps(vars(args), indent=4)
        with open(os.path.join(output_dir, "result.txt"), "w") as fout:
            fout.write(ret_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_type", type=str)
    parser.add_argument("--with_cot", action="store_true")
    parser.add_argument("--sample", type=int, default=-1) # -1 means all
    parser.add_argument("--augment_model", type=str, default=None)  
    parser.add_argument("--num_train_epochs", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--inference_method", type=str, required=True, choices=["icl", "dyprag", "dyprag_combine"])
    # DyPRAG
    parser.add_argument("--projector_p", type=int, required=True)
    parser.add_argument("--inference_epoch", type=int, required=True)
    parser.add_argument("--projector_path", type=str, required=True)
    # LoRA
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--rate", type=float)
    args = parser.parse_args()
    assert args.lora_rank and args.lora_alpha, "No Config for LoRA"
    if args.augment_model is None:
        args.augment_model = args.model_name
    print(args)
    main(args)