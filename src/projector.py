# Implementation of Parameter Translator
# Adopted from DyPRAG:https://github.com/Trae1ounG/DyPRAG
from collections import defaultdict
import torch
import torch.nn as nn
from torch.nn import functional as F

class ParameterTranslator(nn.Module):
    def __init__(self, module_list: list[str], layer_idx: list[int], input_dim: int, output_dim: int, lora_rank: int, hidden_dim: int=32):
        super().__init__()
        self.module_list = module_list
        self.layer_idx = layer_idx
        self.projector = nn.ModuleDict()
        for module_name in self.module_list:
            for layer_idx in self.layer_idx:
                self.projector[f"{module_name}_{layer_idx}"] = Projector(module_name, layer_idx, input_dim, output_dim,  lora_rank, hidden_dim)
    
    def forward(self, x):
        ret = defaultdict(list)
        for module_name in self.module_list:
            for layer_idx in self.layer_idx:
                lora_A, lora_B = self.projector[f"{module_name}_{layer_idx}"](x)
                ret[f"base_model.model.model.layers.{layer_idx}.mlp.{module_name}.lora_A.weight"] = lora_A
                ret[f"base_model.model.model.layers.{layer_idx}.mlp.{module_name}.lora_B.weight"] = lora_B
        return ret

class Projector(nn.Module):
    def __init__(self, module_name, layer_idx, input_dim, output_dim, lora_rank, hidden_dim=8):
        super().__init__()
        self.module_name = module_name
        self.layer_idx = layer_idx
        self.projector = ProjectorLoRA(module_name, layer_idx, input_dim, lora_rank, output_dim, hidden_dim)
    
    def forward(self, x):
        idxs_tensor = torch.tensor(self.layer_idx, device=x.device, dtype=torch.float32).view(-1, 1)
        network_input = torch.cat([x, idxs_tensor], dim=1) # concatenate
        self.lora_A = self.projector.A_hypernet(network_input)
        self.lora_B = self.projector.B_hypernet(network_input)
        return self.lora_A, self.lora_B            
           

class ProjectorLoRA(nn.Module):
    def __init__(self, module_name, layer_idx, input_dim, lora_rank, output_dim, hidden_dim=16):
        super().__init__()
        self.module_name = module_name
        self.layer_idx = layer_idx
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        #Initialize the weight of the LoRA A projector
        self.pre_A_linear = nn.Linear(input_dim + 1, hidden_dim, bias=False, dtype=torch.float32)
        self.pre_A_linear.weight = self.init_layer(self.pre_A_linear)
        if self.module_name == "down_proj":
            self.post_A_linear = nn.Linear(hidden_dim, output_dim * lora_rank, bias=False, dtype=torch.float32)
        else:
            self.post_A_linear = nn.Linear(hidden_dim, input_dim * lora_rank, bias=False, dtype=torch.float32)
        self.post_A_linear.weight = self.init_layer(self.post_A_linear)
        if self.module_name == "down_proj":
            self.A_hypernet = MLPHypernet(self.pre_A_linear, self.post_A_linear, lora_rank, output_dim)
        else:
            self.A_hypernet = MLPHypernet(self.pre_A_linear, self.post_A_linear, lora_rank, input_dim)
        
        #Initialize the weight of the LoRA B projector
        self.pre_B_linear = nn.Linear(input_dim + 1, hidden_dim, bias=False, dtype=torch.float32)
        self.pre_B_linear.weight = self.init_layer(self.pre_B_linear)
        if self.module_name == "down_proj": 
            self.post_B_linear = nn.Linear(hidden_dim, input_dim * lora_rank, bias=False, dtype=torch.float32)
        else:
            self.post_B_linear = nn.Linear(hidden_dim, output_dim * lora_rank, bias=False, dtype=torch.float32)
        self.post_B_linear.weight = self.init_layer(self.post_B_linear)
        if self.module_name == "down_proj":
            self.B_hypernet = MLPHypernet(self.pre_B_linear, self.post_B_linear, input_dim, lora_rank)
        else:
            self.B_hypernet = MLPHypernet(self.pre_B_linear, self.post_B_linear, output_dim, lora_rank)
        
    def init_layer(self, layer):
        weight = nn.Parameter(torch.normal(0, 1e-7, layer.weight.shape))
        return weight
    
        

class MLPHypernet(nn.Module):
    def __init__(self, linear1, linear2, input_dim, output_dim):
        super().__init__()
        self.linear1 = linear1
        self.linear2 = linear2 # 2 layers
        self.input_dim = input_dim  
        self.output_dim = output_dim  
    def forward(self, features):
        # linear1 -> ReLU -> linear2 -> reshape
        output = self.linear2(F.relu(self.linear1(features))).reshape(self.input_dim, self.output_dim)
        return output