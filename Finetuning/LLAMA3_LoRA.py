import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, TrainingArguments 
from datasets import load_dataset 
from peft import get_peft_model, LoraConfig 
import torch 
from trl import SFTTrainer
import math

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
    option_as = examples["opa"]
    option_bs = examples["opb"]
    option_cs = examples["opc"]
    option_ds = examples["opd"]
    answers = examples["cop"]
    explainations = examples["exp"]

    texts = []
    
    for question, option_a, option_b, option_c, option_d, answer, explaination in zip(questions, option_as, option_bs, option_cs, option_ds, answers, explainations):
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
            
# Evaluate the model
def evaluate_model(trainer):
    eval_results = trainer.evaluate()
    return f"Perplexity: {math.exp(eval_results["eval_loss"]):.2f}"
            
def main():
    warnings.filterwarnings('ignore') # Ignore warnings when display the output
    
    # CONFIGURATION
    
    # LoRA config (adapter)
    config = LoraConfig(
        r = 1,
        lora_alpha=32,
        lora_dropout=0.05, #kind of like a regularization dropout
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Config arguments for the training process
    training_args = TrainingArguments(
            learning_rate = 2e-4, # Learning rate change
            lr_scheduler_type = "linear", # Control learning rate change
            eval_strategy= "steps", # Evaluate every 100
            eval_steps= 20,
            warmup_steps = 5,
            weight_decay = 0.01,
            save_strategy= "steps",
            save_steps= 40,
            logging_steps = 1,
            load_best_model_at_end= True,
            gradient_accumulation_steps = 4, # Accumulate gradients for larger batch size
            eval_accumulation_steps= 4, # Accumulate evaluation results for larger batch size
            per_device_eval_batch_size= 1, # Batch size for evaluation
            per_device_train_batch_size= 1, # Batch size per GPU (1 batch contain 1000 data points)
            max_steps = 500,
            seed = 3407,
            fp16 = True, # Use mixed precision training for faster training
            optim = "adamw_8bit", # Use 8-bit optimization for faster training
            group_by_length = True, # Group samples of same length to reduce padding and speed up training
            output_dir = "Finetuning/Fine-tuned_checkpoint/medical_3",
        )
    
    # LOADDING
    
    # Load device
    def get_device_map() -> str:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    device = get_device_map()

    # Load base model
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype=torch.bfloat16)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    EOS_TOKEN = tokenizer.eos_token # End of sequence token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load the dataset
    data_files = {
        "train" : "../medical_3/train.csv",
        "validation": "../medical_3/validation.csv",
        "test" : "../medical_3/test.csv"
    }
    
    dataset = load_dataset("csv", data_files=data_files)
    
    # sample the dataset
    validation_subset = dataset['validation'].train_test_split(test_size=0.1, seed=3407)
    # train_subset = dataset['train'].train_test_split(test_size=0.2, seed=3407)
    
    # IMPLEMENTING LORA TECHNIQUE
    
    # Freezing the original weights
    freeze_model(model)
    
    # Integrate the adapter with the base model
    model = get_peft_model(model, config)

    # Print the trainable parameters
    print_trainable_parameters(model)
    
    # DATA PREPROCESSING AND TOKENIZING
    
    # Create the prompt
    prompt = "You are an assistant for question-answering tasks.\nBelow is the instruction. Answer the question and explain your answer.\nIf you don't know the answer or explaination, just say you don't know."
    
    # Tokenize the dataset
    
    tokenized_dataset = dataset.map(tokenize_function, fn_kwargs= {"prompt": prompt, "EOS_TOKEN": EOS_TOKEN} , batched=True)
    tokenized_validation_subset = validation_subset.map(tokenize_function, fn_kwargs= {"prompt": prompt, "EOS_TOKEN": EOS_TOKEN}, batched=True)
    # tokenized_train_subset = train_subset.map(tokenize_function, fn_kwargs= {"prompt": prompt, "EOS_TOKEN": EOS_TOKEN}, batched=True)
    
    # TRAINING
    
    # Training setup
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = tokenized_dataset['train'],
        eval_dataset= tokenized_validation_subset['test'],
        dataset_text_field = "text",
        max_seq_length = 512,
        packing = False, # Can make training 5x faster for short sequences.
        args = training_args,
    )
    
    # EVALUATING
    questions = ["### Question:\nChronic urethral obstruction due to benign prismatic hyperplasia can lead to the following change in kidney parenchyma\n###Options: \nA. Hyperplasia\nB. Hyperophy.\nC. Atrophy\nD. Dyplasia"]
    
    # Evaluate the base model
    
    # print("Base model predictions:")
    # for question in questions:
    #     print(generate_output(model, tokenizer, question, prompt))
    
    # print(evaluate_model(trainer)) # Evaluate using perplexity
    
    # Start training
    trainer.train()
    
    # Evaluate the fine-tuned model
    
    print("Fine-tuned model predictions:")
    for question in questions:
        print(generate_output(model, tokenizer, question, prompt))
    
    print(evaluate_model(trainer)) # Evaluate using perplexity
        
    # Save the model
    model.save_pretrained("Finetuning/medical_3_LLAMA3_Fine-tuned")

if __name__ == "__main__":
    
    main()