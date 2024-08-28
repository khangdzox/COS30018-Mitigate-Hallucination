import warnings

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, DataCollatorWithPadding, Trainer, TrainingArguments

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
    data_files = {
        "train": "Finetuning/Dataset/medical_1/english-train.json",
        "test": "Finetuning/Dataset/medical_1/english-test.json",
        "validation": "Finetuning/Dataset/medical_1/english-dev.json"
    }
    raw_dataset = load_dataset("json", data_files=data_files)
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)
    
    # Training hyperparameters
    training_args = TrainingArguments(output_dir="test_trainer")
    
    # evaluation
    metric = evaluate.load("accuracy")
    
    def compute_metric(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    #training
    trainer = Trainer(model = model, args = training_args, train_dataset= tokenized_datasets["train"], eval_dataset=tokenized_datasets["validation"], compute_metrics=compute_metric)
    trainer.train()



if __name__ == "__main__":
    main()
