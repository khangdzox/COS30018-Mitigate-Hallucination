import warnings

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from datasets import load_dataset, DatasetDict, Dataset

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig

import torch
import numpy as np
import evaluate

def main():
    warnings.filterwarnings('ignore') # Ignore warnings when display the output
    
    
    # load base model
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
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
    
    #generation config
    generation_config = {
        "max_new_tokens": 200,
        "_from_model_config ": True,
        "bos_token_id": 1,
        "eos_token_id": 11,
        "pad_token_id": 11,
        "temperature": 0.7,
        "top_p": 0.7,
    }
    
    #prompt
    prompt = f"""
    <human> Who is at risk for Lymphocytic Choriomeningitis (LCM)? ?
    <assistant>:
    """.strip()
    
    encoding = tokenizer(prompt, return_tensors="pt")
    with torch.interface_mode():
        outputs = model.generate(input_ids=encoding.input_ids, attention_mask=encoding.attention_mask, generation_config=generation_config)
    
    # Load the dataset
    dataset = load_dataset("csv", data_files="FineTuning/Dataset/medical_2/medDataset_processed.csv")
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return f"""
    <human> {examples["Question"]}
    <assistant>: {examples["Answer"]}
    """.strip()
    
    def generate_and_tokenize_prompt(examples):
        full_prompt = tokenize_function(examples)
        tokenized_full_prompt = tokenizer(full_prompt, padding=True, truncation=True)
        return tokenized_full_prompt
    
    tokenized_datasets = dataset.map(generate_and_tokenize_prompt)
    
    # Training hyperparameters
    training_args = TrainingArguments(output_dir="test_trainer")
    
    #training
    trainer = Trainer(model = model, args = training_args, train_dataset= tokenized_datasets, data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    trainer.train()



if __name__ == "__main__":
    main()