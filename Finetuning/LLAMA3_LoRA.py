import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, TrainingArguments 
from datasets import load_dataset 
from peft import get_peft_model, LoraConfig 
import torch 
from trl import SFTTrainer

# Show the number of trainable parameters
def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Trainable parameters: {trainable_params} || all params: {all_params} || trainable %: {100 * trainable_params/all_params}" )
    
# Interface to interact with the model
def generate_output(model, tokenizer):
    streamer = TextStreamer(tokenizer) # type: ignore

    input_text = "### Question: What is (are) Parasites - Scabies? \n ### Answer: "
    input_tokens = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.cuda.amp.autocast():
        output_tokens = model.generate(**input_tokens, streamer=streamer, max_new_tokens=100, do_sample=True, top_p=0.8)
    return output_tokens

# Tokenize and formating
def tokenize_function(examples: dict, alpaca_prompt: str, EOS_TOKEN: str):
    inputs = examples["Question"]
    outputs = examples["Answer"]

    texts = []
    
    for input, output in zip(inputs, outputs):
        text = alpaca_prompt + "\n" + "### Question: " +  input + "\n","### Answer: " +  output + EOS_TOKEN
        texts.append(text)
        
    return {"text": texts}

# Freezing the original weights
def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False #freeze the model - train adapters later
        
        if param.ndim == 1:
            param.data = param.data.to(torch.float32) #cast the small parameters (layernorm) to fp32 fpr stability
            
def main():
    warnings.filterwarnings('ignore') # Ignore warnings when display the output
    
    # CONFIGURATION
    
    # LoRA config (adapter)
    config = LoraConfig(
        r = 1,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Config arguments for the training process
    training_args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60,
            learning_rate = 2e-4,
            fp16 = True,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear", # control learning rate change
            seed = 3407,
            output_dir = "LLaMA-3-8B-Instruct-Fine-Tuned-LoRA/medical_2",
        )
    
    # LOADDING
    
    # Load device
    def get_device_map() -> str:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    device = get_device_map()

    # load base model
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype=torch.bfloat16)
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    EOS_TOKEN = tokenizer.eos_token # End of sequence token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load the dataset
    data_file = "Finetuning/Dataset/medical_2/4K_answer_medDataset.csv"
    
    train_dataset= load_dataset('csv', data_files=data_file, split='train')
    
    # IMPLEMENTING LORA TECHNIQUE
    
    # Freezing the original weights
    freeze_model(model)
    
    # Integrate the adapter with the base model
    model = get_peft_model(model, config)

    # Print the trainable parameters
    print_trainable_parameters(model)
    
    # DATA PREPROCESSING AND TOKENIZING
    
    # Create the prompt
    alpaca_prompt = "Below is  a question. Answer that question appropriately."
    
    # Tokenize the dataset
    
    tokenized_dataset = train_dataset.map(tokenize_function, fn_kwargs= {"alpaca_prompt": alpaca_prompt, "EOS_TOKEN": EOS_TOKEN} , batched=True)
    
    # TRAINING
    
    #training setup
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = tokenized_dataset,
        dataset_text_field = "text",
        dataset_num_proc = 2,
        max_seq_length = 1024,
        packing = False, # Can make training 5x faster for short sequences.
        args = training_args,
    )
    
    # Start training
    trainer.train()
    
    # generate output
    print(generate_output(model, tokenizer))

if __name__ == "__main__":
    
    main()