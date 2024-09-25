import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, TrainingArguments 
from datasets import load_dataset 
from peft import get_peft_model, LoraConfig 
import torch 
from trl import SFTTrainer
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

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
    option_as = examples["opa"]
    option_bs = examples["opb"]
    option_cs = examples["opc"]
    option_ds = examples["opd"]
    answers = examples["cop"]
    explainations = examples["exp"]
    subjects = examples["subject_name"]

    texts = []
    
    for question, option_a, option_b, option_c, option_d, answer, explaination, subject in zip(questions, option_as, option_bs, option_cs, option_ds, answers, explainations, subjects):
        text = f"""
        {prompt}
        
        ### Question:
        {question}
        
        ### Subject:
        {subject}
        
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
        r = 8,
        lora_alpha=32,
        lora_dropout=0.05, #kind of like a regularization dropout
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Config arguments for the training process
    training_args = TrainingArguments(
            learning_rate = 2e-4, # Learning rate change
            lr_scheduler_type = "cosine", # Control learning rate change
            warmup_ratio = 0.03,
            weight_decay = 0.01,
            save_strategy= "steps",
            save_steps= 10,
            logging_steps = 1,
            gradient_accumulation_steps = 2, # Accumulate gradients for larger batch size
            per_device_train_batch_size= 1, # Batch size per GPU (1 batch contain 1000 data points)
            max_steps = 110,
            seed = 3407,
            fp16 = True, # Use mixed precision training for faster training
            optim = "adamw_8bit", # Use 8-bit optimization for faster training
            group_by_length = True, # Group samples of same length to reduce padding and speed up training
            output_dir = "Finetuning/Fine-tuned_checkpoint/medical_3/6",
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
        "train" : "../medical_3/clean_train.csv",
        "validation": "../medical_3/clean_validation.csv",
        "test" : "../medical_3/test.csv"
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
    prompt = """
You are an assistant for question-answering tasks.
First check user input, if it is only question, then give the answer follow this format:
### Answer: <answer>

### Explaination: <explaination>

The explaination should be detailed and clear.

if the user input is a question with options, then give the answer follow this format:

### Answer: <correct option>

### Explaination: <explaination>

The explaination should explain why the other options are incorrect and why the correct option is correct.

If you don't know the answer or explaination, just say you don't know.

   Here is some example questions and answer:
    ### Example 1:
    ### Question:
    Which of the following is an example for reversible dementia?
    
    ### Subject:
    Psychiatry
    
    ### Options:
    A. Normal pressure hydrocephalus
    B. Alzheimer's dementia
    C. Lewy body dementia
    D. CreutzFeldt Jakob disease
    
    ### Answer:
    A
    
    ### Explaination:
    Impoant possibly reversible conditions are: Substance and medication related Anticholinergics, anti-hypeensives, sedative hypnotics Psychiatric disorders Depression Metabolic and endocrinal disorders Hypothyroidism, Vitamin B12 deficiency, Hepatic and Renal failure Neurosurgical conditions Normal pressure Hydrocephalus, Brain tumor, Subdural hematoma Neuroinfections Herpes encephalitis Miscellaneous Significant sensory deficits
    
    ### Example 2:
    ### Question:
    A 16 year old female patient presents to the OPD with hirsutism and masculinization. Which of the following hormones of the adrenal coex is the likely culprit?
    
    ### Subject:
    Physiology
    
    ### Answer:
    Dehydroepiandrosterone (DHEA)
    
    ### Explaination:
    Hirsutism and musculanisation in a female suggests excessive androgens like dehydroepiandrosteronen(DHEA), which is culprit here. Adrenogenital sydrome: An adrenocoical tumor secretes excessive quantities of androgens that cause intense masculanizing effects. In women, virile characteristics develop, including growth of a beard, deeper voice, masculine distribution of hair on the body and the pubis and growth of the clitoris. In boys, it presents as precocious pubey. The excretion of 17-ketosteroids (which are derived from androgens) in the urine may be 10 to 15 times elevated. This findings can be used in diagnosing the disease. Ref: Guyton and Hall 13th edition Pgno: 981
    """
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(tokenize_function, fn_kwargs= {"prompt": prompt, "EOS_TOKEN": EOS_TOKEN} , batched=True)
    
    # Limit token number
    # print(len(tokenized_dataset["train"]))
    
    # After filter
    # print(len(tokenized_dataset["train"]))
    
    # Visualize token number
    # visualize_token_lengths(tokenized_dataset["train"])
    
    # shuffle the dataset
    tokenized_dataset['train'] = tokenized_dataset['train'].shuffle(seed=3407)
    tokenized_dataset['validation'] = tokenized_dataset['validation'].shuffle(seed=3407)
    
    # TRAINING
    
    # Training setup
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = tokenized_dataset['train'],
        eval_dataset= tokenized_dataset['validation'],
        dataset_text_field = "text",
        packing = False, # Can make training 5x faster for short sequences.
        args = training_args,
        max_seq_length= 1024,
        dataset_batch_size = 1000,
    )
    
    # EVALUATING
    questions = """
### Question:\nChronic urethral obstruction due to benign prismatic hyperplasia can lead to the following change in kidney parenchyma
    """
    
    # Evaluate the base model
    
    # print("Base model predictions:")
    # for question in questions:
    #     print(generate_output(model, tokenizer, question, prompt))
    
    # Start training
    trainer.train()
    
if __name__ == "__main__":
    
    main()