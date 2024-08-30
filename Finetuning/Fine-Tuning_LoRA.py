import warnings

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from datasets import load_dataset, DatasetDict, Dataset

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig

import torch
import numpy as np
import evaluate

def main():
    warnings.filterwarnings('ignore') # Ignore warnings when display the output
    
    def get_device_map() -> str:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    device = get_device_map()

    # load base model
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype=torch.bfloat16)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
    
    # Freezing the original weights
    for param in model.parameters():
        param.requires_grad = False #freeze the model - train adapters later
        
        if param.ndim == 1:
            param.data = param.data.to(torch.float32) #cast the small parameters (layernorm) to fp32 fpr stability
    
    model.gradient_checkpointing_enable() # reduce number of stored activations
    model.enable_input_require_grads()
    
    #Setting up the LoRA adapter
    def print_trainable_parameters(model):
        trainable_params = 0
        all_params = 0
        
        for _, param in model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"Trainable parameters: {trainable_params} || all params: {all_params} || trainable %: {100 * trainable_params/all_params}" )
    
    # LoRA config
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
    dataset = load_dataset("mlabonne/guanaco-llama2-1k", split="train")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"])
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"]
    )
    
    # Training hyperparameters
    training_args = TrainingArguments(output_dir="test_trainer")
    
    #training
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
    )
    trainer.train()



if __name__ == "__main__":
    main()