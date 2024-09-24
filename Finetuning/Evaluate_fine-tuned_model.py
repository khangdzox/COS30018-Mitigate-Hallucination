import warnings
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import torch
import sys
from datasets import load_dataset 
import math
from trl import SFTTrainer


# Add the Finetuning directory to the Python path
sys.path.append('Finetuning')

from LLAMA3_LoRA import generate_output, tokenize_function # type: ignore

def main():
    warnings.filterwarnings('ignore') # Ignore warnings when display the output
    
    # CONFIGURATION
    training_args = TrainingArguments(
            per_device_train_batch_size= 1, # Batch size per GPU (1 batch contain 1000 data points)
            per_device_eval_batch_size= 1, # Batch size for evaluation
            gradient_accumulation_steps = 4, # Accumulate gradients for larger batch size
            eval_accumulation_steps= 4, # Accumulate evaluation results for larger batch size
            eval_strategy= "steps", # Evaluate every 100
            warmup_steps = 5,
            logging_steps = 1,
            learning_rate = 1e-4, # Learning rate change
            fp16 = True, # Use mixed precision training for faster training
            optim = "adamw_8bit", # Use 8-bit optimization for faster training
            weight_decay = 0.01,
            lr_scheduler_type = "linear", # Control learning rate change
            seed = 3407,
            output_dir = "LLaMA-3-8B-Instruct-Fine-Tuned-LoRA/medical_3",
            group_by_length = True, # Group samples of same length to reduce padding and speed up training
            max_steps = 200,
            eval_steps= 20,
        )
    
    lora_config = PeftConfig.from_pretrained("./Finetuning/medical_3_LLAMA3_Fine-tuned/3")
    
    # LOADDING
    
    # Load base model
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    
    # Load Pre-trained model
    model = PeftModel.from_pretrained(base_model, "./Finetuning/medical_3_LLAMA3_Fine-tuned/3", device_map = "cuda", torch_dtype = torch.bfloat16)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    EOS_TOKEN = tokenizer.eos_token # End of sequence token
    tokenizer.pad_token = tokenizer.eos_token

    # Load the dataset
    data_path = {
        'test': "../medical_3/clean_validation.csv"
        }
    
    # dataset = load_dataset("csv", data_files=data_path)

    # GENERATE OUTPUT
    prompt = "You are an assistant for question-answering tasks.\nBelow is the instruction. Answer the question and explain your answer.\nIf you don't know the answer or explaination, just say you don't know."
    question = "### Question:\nChronic urethral obstruction due to benign prismatic hyperplasia can lead to the following change in kidney parenchyma\n###Options: \nA. Hyperplasia\nB. Hyperophy.\nC. Atrophy\nD. Dyplasia"
 
    print(generate_output(model, tokenizer, question, prompt))
    
    # Tokenize the dataset
    # tokenized_validation = dataset.map(tokenize_function, fn_kwargs= {"prompt": prompt, "EOS_TOKEN": EOS_TOKEN} , batched=True)
    
    # Evaluate model
    # trainer = SFTTrainer(
    #     model = model,
    #     tokenizer = tokenizer,
    #     dataset_text_field = "text",
    #     max_seq_length = 512,
    #     eval_dataset = tokenized_validation,
    #     args = training_args,
    # )

if __name__ == "__main__":
    main()
