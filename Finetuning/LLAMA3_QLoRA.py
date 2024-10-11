import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset 
from peft import get_peft_model, LoraConfig 
import torch 
from trl import SFTTrainer
from datasets import load_metric
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

import matplotlib.pyplot as plt

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

# Define the Exact Match metric
def compute_exact_match(predictions, references):
    metric = load_metric("exact_match")
    return metric.compute(predictions=predictions, references=references)

# Filter tokenized_data set
def filter_max_tokens(example, max_tokens):
    return len(example['text']) <= max_tokens

# Tokenize visualization
def visualize_token_lengths(dataset):
    token_lengths = [len(example["text"]) for example in dataset]
    
    plt.hist(token_lengths, bins=50, edgecolor='black')
    plt.title('Token Length Distribution')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.show()

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
def generate_output(model, tokenizer, question, prompt):
    streamer = TextStreamer(tokenizer) # Type: ignore

    input_text = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question},
    ]
    input_tokens = tokenizer.apply_chat_template(input_text, return_tensors="pt").to(model.device)
    
    with torch.cuda.amp.autocast(): # Make sure the model and input are in the same fp16 format
        output_tokens = model.generate(input_tokens, streamer=streamer, max_new_tokens=512, do_sample=True, top_p=0.8, pad_token_id=tokenizer.eos_token_id)
    return output_tokens

# Tokenize and formating
def tokenize_function(examples, prompt, EOS_TOKEN):
    questions = examples["question"]
    option_as = examples["option_a"]
    option_bs = examples["option_b"]
    option_cs = examples["option_c"]
    option_ds = examples["option_d"]
    answers = examples["cop"]
    explainations = examples["exp"]

    texts = []
    
    for question, option_a, option_b, option_c, option_d, answer, explaination,in zip(questions, option_as, option_bs, option_cs, option_ds, answers, explainations):
        text = f"""
        {prompt}
        
        ### Question:
        {question}
        
        ### Options:
        A. {option_a}
        B. {option_b}
        C. {option_c}
        D. {option_d}
        
        ### Answer:
        {answer}
        
        ### Explaination:
        {explaination}
        {EOS_TOKEN}
        """
        texts.append(text)
    
    return {"text": texts}

# Freezing the original weights
def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False # Freeze the model - train adapters later
        
        if param.ndim == 1:
            param.data = param.data.to(torch.float32) # Cast the small parameters (layernorm) to fp32 fpr stability
            
def main():
    warnings.filterwarnings('ignore') # Ignore warnings when display the output
    
    # CONFIGURATION
    
    # LoRA config (adapter)
    config = LoraConfig(
        r = 2,
        lora_alpha=32,
        lora_dropout=0.05, #kind of like a regularization dropout
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj"]
    )
    
    # Config arguments for the training process
    training_args = TrainingArguments(
            learning_rate = 2e-4, # Learning rate change 
            lr_scheduler_type = "cosine", # Control learning rate change
            lr_scheduler_kwargs= {"num_cycles": 2},
            warmup_ratio= 0.05,
            weight_decay = 0.01,
            save_strategy= "steps",
            save_steps= 10,
            eval_steps= 50,
            eval_strategy= "steps",
            logging_steps= 1,
            gradient_accumulation_steps = 16, # Accumulate gradients for larger batch size
            eval_accumulation_steps= 32,
            per_device_train_batch_size= 1, # Batch size per GPU 
            per_device_eval_batch_size= 2,
            max_steps = 500,
            seed = 3407,
            fp16 = True, # Use mixed precision training for faster training
            optim = "adamw_torch",
            # group_by_length = True, # Group samples of same length to reduce padding and speed up training
            output_dir = "Finetuning/Fine-tuned_checkpoint/medical_3/QLoRA/8",
            max_grad_norm= 1.0  # Apply gradient clipping
        )
    
    # LOADDING
    
    # Load device
    def get_device_map() -> str:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # device = get_device_map()

    # Load base model
    
    # Quantization configuration
    
    # Activate 4-bit precision base model loading
    use_4bit = True

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"

    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = True
    
    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.bfloat16, quantization_config=bnb_config,)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    EOS_TOKEN = tokenizer.eos_token # End of sequence token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load the dataset
    data_files = {
        "train" : "../medical_3/final_train.csv",
    }
    
    dataset = load_dataset("csv", data_files=data_files)
    
    # IMPLEMENTING LORA TECHNIQUE
    
    # Freezing the original weights
    freeze_model(model)
    
    # Integrate the adapter with the base model
    model = get_peft_model(model, config)

    # Print the trainable parameters
    print_trainable_parameters(model)
    
    # DATA PREPROCESSING AND TOKENIZING
    
    # Create the prompt
    prompt = "You are an assistant for question-answering tasks. Answering and explaining the questions appropriately."
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(tokenize_function, fn_kwargs= {"prompt": prompt, "EOS_TOKEN": EOS_TOKEN} , batched=True)
    
    # Limit token number
    filtered_tokenized_dataset = tokenized_dataset['train'].filter(filter_max_tokens, fn_kwargs={"max_tokens": 1024})
    
    filtered_tokenized_dataset = filtered_tokenized_dataset.train_test_split(test_size=0.05, shuffle=True)
    
    # print(filtered_tokenized_dataset['text'][0])
    # Visualize token number
    # print(len(filtered_tokenized_dataset['train']))
    # visualize_token_lengths(filtered_tokenized_dataset['train'])
    
    # TRAINING
    
    # Training setup
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = filtered_tokenized_dataset['train'],
        eval_dataset= filtered_tokenized_dataset['test'],
        dataset_text_field = "text",
        # packing = False, # Can make training 5x faster for short sequences.
        args = training_args,
        max_seq_length= 1024,
        dataset_batch_size= 2048,
    )
    
    # EVALUATING
    questions = "Chronic urethral obstruction due to benign prismatic hyperplasia can lead to the following change in kidney parenchyma"
    
    # Evaluate the base model
    
    # print("Base model predictions:")
    # for question in questions:
    #     print(generate_output(model, tokenizer, question, prompt))
    
    # Start training
    # trainer.train()
    
if __name__ == "__main__":
    
    main()