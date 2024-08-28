import warnings

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, DataCollatorWithPadding, Trainer, TrainingArguments

from datasets import load_dataset, DatasetDict, Dataset

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig

import torch
import numpy as np

def main():
    warnings.filterwarnings('ignore') # Ignore warnings when display the output
    
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Freezing the original weights
    for param in model.parameters():
        param.requires_grad = False #freeze the model - train adapters later
        
        if param.ndim == 1:
            param.data = param.data.to(torch.float32) #cast the small parameters (layernorm) to fp32 fpr stability
    
    model.gradient_checkpointing_enable() # reduce number of stored activations
    model.enable_input_require_grads()
    
    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float32)
    model.lm_head = CastOutputToFloat(model.lm_head)    
    
    #Setting up the LoRA adapter
    def print_trainable_parameters(model):
        trainable_params = 0
        all_params = 0
        
        for _, param in model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"Trainable parameters: {trainable_params} || all params: {all_params} || trainable %: {100 * trainable_params}" )
    
    config = LoraConfig(
        r = 16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    
    # Load the dataset
    data = load_dataset("lavita/medical-qa-shared-task-v1-toy")



if __name__ == "__main__":
    main()
