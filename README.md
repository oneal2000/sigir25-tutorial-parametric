# Parametric RAG Toolkit

Welcome to the **Parametric RAG Toolkit**, developed as part of our **SIGIR 2025 Tutorial: Dynamic and Parametric Retrieval-Augmented Generation**.

This repository provides a comprehensive and easy-to-use toolkit designed to help researchers and practitioners quickly reproduce, compare, and extend **Parametric Retrieval-Augmented Generation (Parametric RAG)** methods, specifically **PRAG** and **DyPRAG**.

⭐️ **Star this repository** to support our work and stay updated!

## Table of Contents

- [Parametric RAG Toolkit](#parametric-rag-toolkit)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [📌 Supported Methods](#-supported-methods)
  - [Quick Start](#quick-start)
    - [1️⃣ Preparations](#1️⃣-preparations)
    - [2️⃣ Clone and Installation](#2️⃣-clone-and-installation)
    - [3️⃣ Data Preparation](#3️⃣-data-preparation)
      - [Prepare BM25 for retrieval](#prepare-bm25-for-retrieval)
      - [Download dataset](#download-dataset)
    - [4️⃣ Run Data Augmentation](#4️⃣-run-data-augmentation)
    - [5️⃣ Run Parametric Knowledge Encoding](#5️⃣-run-parametric-knowledge-encoding)
    - [6️⃣  Train Parameter Translator in DyPRAG](#6️⃣--train-parameter-translator-in-dyprag)
    - [7️⃣  Inference with Parametric Knowledge](#7️⃣--inference-with-parametric-knowledge)
  - [Toolkit Structure](#toolkit-structure)
  - [Customize and Extend](#customize-and-extend)
    - [🔄 Switch Base LLM Models](#-switch-base-llm-models)
    - [🗂️ Add New Datasets](#️-add-new-datasets)
  - [Detailed Usage Guide](#detailed-usage-guide)
    - [⚙️ Parametric Knowledge Encoding (Offline)](#️-parametric-knowledge-encoding-offline)
    - [🧠 Parametric Knowledge Inferencing (Online)](#-parametric-knowledge-inferencing-online)
  - [🤝 Contributing](#-contributing)
  - [📜 Citation](#-citation)


## Overview

The Parametric RAG Toolkit simplifies experimenting with Parametric RAG techniques, a powerful approach to retrieval-augmented generation by encoding external knowledge into model parameters using LoRA (Low-Rank Adaptation). This toolkit enables users to:

* **Reproduce PRAG and DyPRAG methods** from end-to-end.
* Easily **switch base LLM models** and **extend to new datasets**.
* Understand how to **generate and utilize LoRA adapters** during offline training and inference stages.

## 📌 Supported Methods

Currently supported:

* ✅ **PRAG** ([SIGIR 2025 Paper](https://github.com/oneal2000/PRAG))
* ✅ **DyPRAG** ([Arxiv Paper](https://arxiv.org/abs/2503.23895), [GitHub](https://github.com/Trae1ounG/DyPRAG))

More Parametric RAG variants will be supported soon!

## Quick Start

Follow these steps and you can quickly run Parametric RAG experiments:  

### 1️⃣ Preparations

**Before you really start to use this toolkit, please make sure you've finished the following preparations:**
- Change the path of [`src/root_dir_path.py`](src/root_dir_path.py) to the path you place this toolkit.  
  For example, if you place this toolkit in `/home/user/sigir25-tutorial-parametric`, you should change the content of [`src/root_dir_path.py`](src/root_dir_path.py) to:
  ```python
  ROOT_DIR = "/home/user/sigir25-tutorial-parametric"
  ```
- If you've downloaded the LLM models manually, you can modify the paths in [`src/utils.py`](src/utils.py) and [`src/retrieve/retriever.py`](src/retrieve/retriever.py) to point to your local model directories.Alternatively, you can use our default settings, which will automatically download the models from HuggingFace if they are not already cached locally.  

### 2️⃣ Clone and Installation

Firstly, you need to clone this repository.
```bash
git clone https://github.com/oneal2000/sigir25-tutorial-parametric.git
cd sigir25-tutorial-parametric
```
Then, you need to install the required dependencies.
```bash
conda create -n prag python=3.10.4
conda activate prag
pip install -r requirements.txt
```

### 3️⃣ Data Preparation

#### Prepare BM25 for retrieval

1. Download the Wikipedia dump from the [DPR repository](https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py#L32) with the script below:

```bash
bash scripts/download_dpr.sh
```

2. Use Elasticsearch to index the Wikipedia dump:

```bash
bash scripts/prep_elastic.sh
```

- **NOTE**: Due to environment differences, there may be some issues with the Elasticsearch setup. Therefore, we strongly reccommend you to use LLMs(ChatGPT, Gemini, etc) to help you resolve errors if you encounter any. Besides, please read the comments in this bash script carefully because some parts are ONLY needed for first use and you should comment them afterwards, for example, the part to download the elasticsearch is only needed for the first time you run the script.  

#### Download dataset

**We provide ways to download 4 types of datasets for you to experiment with, including 2WikiMultihopQA, HotpotQA, PopQA, and ComplexWebQuestions. To reproduce the results in this toolkit, you just need to download popQA and ComplexWebQuestions datasets. You can download them by running the corresponding commands below.**

For 2WikiMultihopQA:

Download the [2WikiMultihopQA](https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?e=1) dataset from its repository <https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?e=1>. Unzip it and move the folder to `data/2wikimultihopqa`.

For HotpotQA:

```bash
bash scripts/download_hotpotqa.sh
```

For PopQA:

Download the [PopQA](https://github.com/AlexTMallen/adaptive-retrieval?tab=readme-ov-file#popqa) dataset from its repository <https://github.com/AlexTMallen/adaptive-retrieval/blob/main/data/popQA.tsv>, and put the file `popQA.tsv` into folder `data/popqa`.  

For ComplexWebQuestions:

Download the [ComplexWebQuestions](https://www.tau-nlp.sites.tau.ac.il/compwebq) dataset from its repository <https://www.dropbox.com/scl/fo/nqujvpg2gc4y0ozkw3wgr/AOzjVEsdUhv2Fx2pamfJlSw?rlkey=746t7xehfqxf1zr867nxiq8aq&e=1>, and put the file `ComplexWebQuestions_dev.json` into folder `data/complexwebquestions`.


### 4️⃣ Run Data Augmentation

**Data augmentation is aimed to integrate multiple rewrites with corresponding QA pairs of a given document to generate a more comprehensive document that consists of diversed linguistic variations.**

For PRAG, you need to run command like this:
```bash
python src/augment.py \
    --model_name llama3.2-1b-instruct \
    --dataset popqa \
    --data_path data/popqa/ \
    --sample 300  \
    --topk 3
```

The results of data augmentation for PRAG will be stored in the file `data_aug/{dataset}/{data_type}.json`. And they will be used to generate parameterize document in PRAG and infernece.  

To reproduce the results showed in this toolkit, you can directly run the script:  

```bash
bash configs/PRAG/augment/augment_prag.sh
```

For training DyPRAG parameter translator, you need to set `output_dir` to `data_aug_projector` and set `projector`.  
```bash
python src/augment.py \
    --model_name llama3.2-1b-instruct \
    --dataset popqa \
    --data_path data/popqa/ \
    --sample 200  \
    --topk 3 \
    --output_dir data_aug_projector \
    --projector \
```
The results of data augmentation will be stored in the file `data_aug_projector/{dataset}/{data_type}.json`. This augmented dataset will be used to train the parameter translator in DyPRAG.    

According to DyPRAG, you should collect 200 additional questions besides the original 300 questions collected in `data_aug`, and use 3 different models to augment the data. Thus, you'll get 4800 samples for the parameter translator training.  

For convinence, we provide pre-augmented data files, which include 4 types of datasets and each dataset is augmented by 3 models, and we recommend you to use them directly, you can use them by running the command:
```bash
tar -xzvf data_aug.tar.gz
```

### 5️⃣ Run Parametric Knowledge Encoding

```bash
python src/encode.py \
    --model_name=llama3.2-1b-instruct \
    --dataset=popqa \
    --sample=300 \
    --per_device_train_batch_size=1 \
    --num_train_epochs=1 \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32 \
    --with_cot \
    --projector
```

For DyPRAG training, set `projector` and for PRAG inference unset `projector`.   

**All running parameters used in encoding in PRAG can be found in `configs/PRAG/encode` and if you want to reproduce the results showed in this toolkit, you can directly run this script:** 

```bash
bash configs/PRAG/encode/encode_prag.sh
```

All running parameters used to get samples for DyPRAG training can be found in `configs/DyPRAG/encode`, if you want to train the parameter translator by yourself, you need to run 12 scripts in `configs/DyPRAG/encode`, which will generate 4800 samples for the parameter translator training.

###  6️⃣  Train Parameter Translator in DyPRAG

```bash
python3 -u src/train_dyprag.py \
    --model_name=llama3.2-1b-instruct \
    --datasets="2wikimultihopqa,popqa,hotpotqa,complexwebquestions" \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32 \
    --max_new_tokens=128 \
    --sample_rate=1 \
    --dyprag_learning_rate=1e-5 \
    --dyprag_train_epochs=1 \
```

The well-trained parameter translator will be saved to `projector/f'{args.model_name}_hidden{args.projector_p}_sample{args.sample_rate}_lr{args.dyprag_learning_rate}` folder.  

For convinence, you can directly use the pre-trained parameter translator provided in the official github repository of DyPRAG, you can download them [here](https://drive.google.com/drive/folders/1FLu3_rMcAMaXfQKQSqf6nALhBv841-ko?usp=drive_link), if you want to reproduce the results showed in this toolkit, you need to put the downloaded llama-1b translator file into the folder `projector/llama3.2-1b-instruct_hidden32_sample1.0_lr1e-05` and rename it to `epoch_0.pt` , and put the downloaded qwen-1.5b translator file into the folder `projector/qwen2.5-1.5b-instruct_hidden32_sample1.0_lr1e-05` and rename it to `epoch_0.pt`.   

###  7️⃣  Inference with Parametric Knowledge

For PRAG, you can infer with this command:

```bash
python3 src/inference.py \
    --model_name=llama3.2-1b-instruct \
    --dataset=popqa \
    --sample=300 \
    --num_train_epochs=2 \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32 \
    --max_new_tokens=20 \
    --inference_method=combine 
```

For DyPRAG, you can infer with this command:

```bash
python3 src/inference_dyprag.py \
    --model_name=llama3.2-1b-instruct \
    --dataset=popqa \
    --sample=-1 \
    --num_train_epochs=1 \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32 \
    --max_new_tokens=128 \
    --inference_method=dyprag \
    --inference_epoch=1 \
    --projector_path="llama3.2-1b-instruct_hidden32_sample1.0_lr1e-05" \
    --projector_p=32
```

- We test 5 ways of inference in this toolkit, including `icl`, `prag`, `prag_combine`, `dyprag`, and `dyprag_combine`. 
- **All running parameters used in inference can be found in `configs/PRAG/inference` and `configs/DyPRAG/inference`, and you can directly run these scripts to reproduce the results.** 
- The inference process will generate three files for each sub-dataset:  
  - `config.json`:the configuration of the inference process, including the model name, dataset, learning rate, etc.
  - `predict.json`: the predicted answer for each question in the dataset and evaluation results like F1 score, EM score for each question.
  - `result.txt`: the overall evaluation results like average F1 score, average EM score, etc.


We conducted experiments on two datasets, PopQA and ComplexWebQuestions, using two LLMs, LLama3.2-1B and Qwen2.5-1.5B in this toolkit. The results are shown in the table below:

| Model         | Method             | popqa     | Script         | complexwebquestions | Script         |
|---------------|--------------------|-----------|----------------|---------------------|----------------|
| LLama3.2-1B   | standard RAG(ICL)  | 0.2025    |        [icl](configs/inference/icl/popqa_llama3.2-1b-instruct.sh)           | 0.3762              |      [icl](configs/inference/icl/complexwebquestions_llama3.2-1b-instruct.sh)       |
|               | PRAG               | 0.2150    |       [prag](configs/inference/prag/popqa_llama3.2-1b-instruct.sh)        | 0.3525              |        [prag](configs/inference/prag/complexwebquestions_llama3.2-1b-instruct.sh)        |
|               | PRAG-combine       | **0.3271**|       [prag_combine](configs/inference/prag_combine/popqa_llama3.2-1b-instruct.sh)         | **0.4024**          |        [prag_combine](configs/inference/prag_combine/complexwebquestions_llama3.2-1b-instruct.sh)        |
|               | DyPRAG             | 0.0937    |        [DyPRAG](configs/inference/dyprag/popqa_llama3.2-1b-instruct.sh)        | 0.3633              |        [DyPRAG](configs/inference/dyprag/complexwebquestions_llama3.2-1b-instruct.sh)        |
|               | DyPRAG-combine     | 0.3144    |       [DyPRAG_combine](configs/inference/dyprag_combine/popqa_llama3.2-1b-instruct.sh)         | 0.3921              |        [DyPRAG_combine](configs/inference/dyprag_combine/complexwebquestions_llama3.2-1b-instruct.sh)        |
| Qwen2.5-1.5B  | standard RAG(ICL)  | 0.0999    |        [icl](configs/inference/icl/popqa_qwen2.5-1.5b-instruct.sh)        | 0.2823              |       [icl](configs/inference/icl/complexwebquestions_qwen2.5-1.5b-instruct.sh)         |
|               | PRAG               | 0.2162    |        [PRAG](configs/inference/prag/popqa_qwen2.5-1.5b-instruct.sh)        | 0.3082              |        [PRAG](configs/inference/prag/complexwebquestions_qwen2.5-1.5b-instruct.sh)        |
|               | PRAG-combine       | **0.2364**|       [PRAG_combine](configs/inference/prag_combine/popqa_qwen2.5-1.5b-instruct.sh)         | 0.3209              |        [PRAG_combine](configs/inference/prag_combine/complexwebquestions_qwen2.5-1.5b-instruct.sh)        |
|               | DyPRAG             | 0.0664    |        [DyPRAG](configs/inference/dyprag/popqa_qwen2.5-1.5b-instruct.sh)        | 0.3194              |       [DyPRAG](configs/inference/dyprag/complexwebquestions_qwen2.5-1.5b-instruct.sh)         |
|               | DyPRAG-combine     | 0.2269    |        [DyPRAG_combine](configs/inference/dyprag_combine/popqa_qwen2.5-1.5b-instruct.sh)        | **0.3357**          |        [DyPRAG_combine](configs/inference/dyprag_combine/complexwebquestions_qwen2.5-1.5b-instruct.sh)        |


All results above are reported as F1 scores, and the best results are highlighted in bold. The running parameters used in each experiment can be found in the corresponding script showed in the table above.



## Toolkit Structure

```
Parametric-RAG-Toolkit/
├── configs/                   # Example configurations for PRAG & DyPRAG
├── data/                      # Data storage and preprocessing scripts
├── scripts/                   # Data download and preparation scripts
├── src/
│   ├── fewshot                # Provide few-shot learning samples
│   ├── retrieve               # Implementation of BM25 retriever
│   ├── models                 # Implementation of parameter injection for LLMs
│   ├── augment.py             # Data augmentation script
│   ├── encode.py              # Generate parametric knowledge (LoRA)
│   ├── train_dyparg.py        # Train the parameter translator for DyPRAG 
│   ├── inference.py           # Inference using parametric knowledge for PRAG
│   ├── inference_dyprag.py    # Inference for DyPRAG
|   ├── projector.py           # Implementation of parameter translator in DyPRAG
|   ├── root_dir_path.py       # The path you place this toolkit
|   ├── prompt_template.py     # Provide prompts' templates for model generation
│   └── utils.py               # Common utilities and evaluation scripts
├── prep_elastic.py            # Build index for wikipedia dump using Elasticsearch
├── requirements.txt           # Python dependencies
├── data_aug.tar.gz            # Pre-augmented data files
└── README.md                  # Documentation and usage guide
```


##  Customize and Extend

The Parametric RAG Toolkit is designed for flexibility and ease of extension.


### 🔄 Switch Base LLM Models

To switch the base LLM:

* Choose your desired LLM from [transformers.models](https://github.com/huggingface/transformers/tree/main/src/transformers/models)
* Copy `configuration_xxx.py` and `modeling_xxx.py` to the `models` folder and modify the import information in`modeling_xxx.py` similar to our [`src/models/modeling_qwen2.py`](#-src/models/modeling_qwen2.py)
* Modify `forward` function of MLP module in `modeling_xxx.py` similar to our [`src/models/modeling_qwen2.py`](#-src/models/modeling_qwen2.py) Line 57-69
* Add a new class in `get_model_class` function in [`src/utils.py`](#-src/utils.py) to load the new type of LLMs.
* Add a new path in `get_model_path` function in [`src/utils.py`](#-src/utils.py) to load the new type of LLMs.
* Update the `--model_name` parameter in scripts and configuration files.




### 🗂️ Add New Datasets

Datasets already supported:
- 2WikiMultihopQA
- HotpotQA
- PopQA
- ComplexWebQuestions




To add a new dataset:

* Prepare your dataset in JSON format with structure:

  ```json
  [
    {
      "question": "your question",
      "answer": "answer text or list of acceptable answers"
    }
  ]
  ```
* Place the file in `data/{your_dataset}`.
* Update data augmentation scripts accordingly:

  ```bash
  python src/augment.py \
    --model_name llama3.2-1b-instruct \
    --dataset your_dataset \
    --data_path data/your_dataset/ \
    --sample 300  \
    --topk 3
  ```

For example, if you want to use StrategyQA dataset, you can download it from [StrategyQA](https://huggingface.co/datasets/voidful/StrategyQA/tree/main) and place it in `data/strategyqa`. Then, you can extract `question` and `answer` from the dataset `strategyqa_train.json` and put them into a json file, for example, `data/strategyqa/total.json`, and then you can run the data augmentation script like this:

```bash
python src/augment.py \
    --model_name llama3.2-1b-instruct \
    --dataset strategyqa \
    --data_path data/strategyqa/ \
    --sample 300  \
    --topk 3
```


##  Detailed Usage Guide

This toolkit divides the process clearly into two stages:

### ⚙️ Parametric Knowledge Encoding (Offline)

* Perform data augmentation to enhance documents.
* Generate LoRA parameters embedding the external knowledge into LLM.
* Train a prameter translator for DyPRAG.

### 🧠 Parametric Knowledge Inferencing (Online)

**PRAG**
* Load pre-generated LoRA parameters.
* Run inference using your customized parametric knowledge.

**DyPRAG**
* Use the trained parameter translator to generate LoRA parameters.
* Run inference using your customized parametric knowledge.


Detailed documentation for each script and parameter can be found within `configs` and `src`.

## 🤝 Contributing

We welcome contributions! Please open an issue or submit a pull request if you want to extend the toolkit or suggest improvements.

## 📜 Citation

If you find this toolkit helpful, please cite our work:

```bibtex
@inproceedings{su2025parametric,
  title={Parametric Retrieval-Augmented Generation},
  author={Su, Weihang and Tang, Yichen and Ai, Qingyao and Yan, Junxi and Wang, Changyue and Wang, Hongning and Ye, Ziyi and Zhou, Yujia and Liu, Yiqun},
  booktitle={SIGIR},
  year={2025}
}
```
🌟 **Thank you for your interest and support!** 🌟