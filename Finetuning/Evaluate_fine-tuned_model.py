import warnings
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer,TextStreamer
import torch
import sys
from datasets import load_dataset 

# Add the Finetuning directory to the Python path
sys.path.append('Finetuning')

def generate_output(model, tokenizer, question):
    streamer = TextStreamer(tokenizer) # Type: ignore

    input_text = {"role": "user", "content": question},
    input_tokens = tokenizer.apply_chat_template(input_text, return_tensors="pt", add_generation_prompt=True).to(model.device)
    
    with torch.cuda.amp.autocast(): # Make sure the model and input are in the same fp16 format
        output_tokens = model.generate(input_tokens, streamer=streamer, max_new_tokens=512, do_sample=True, top_p=0.8, pad_token_id=tokenizer.eos_token_id)
    return output_tokens

def main():
    warnings.filterwarnings('ignore') # Ignore warnings when display the output
 
    # LOADDING
    
    # Load base model
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    
    # Load Pre-trained model
    model = PeftModel.from_pretrained(base_model, "./Finetuning/QLoRA/1", device_map = "cuda", torch_dtype = torch.bfloat16)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    EOS_TOKEN = tokenizer.eos_token # End of sequence token
    tokenizer.pad_token = tokenizer.eos_token

    # GENERATE OUTPUT
    question = "Chronic urethral obstruction due to benign prismatic hyperplasia can lead to the following change in kidney parenchyma"
 
    print(generate_output(model, tokenizer, question))
    

if __name__ == "__main__":
    main()
