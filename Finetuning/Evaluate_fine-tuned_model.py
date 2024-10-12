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
    
    with torch.cuda.amp.autocast(): # Make sure the model and input are in the same fp16 format
        output_tokens = model.generate(tokenizer.encode(question, return_tensors = "pt").to(model.device), streamer=streamer, max_new_tokens=100, do_sample=True, top_p=0.8, pad_token_id=tokenizer.eos_token_id)
    return output_tokens

def main():
    warnings.filterwarnings('ignore') # Ignore warnings when display the output
 
    # LOADDING
    
    # Load device
    def get_device_map() -> str:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
        
    device = get_device_map()
    
    # Load base model
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    
    # Load Pre-trained model
    model = PeftModel.from_pretrained(base_model, "./Finetuning/LoRA/10/Best model", device_map = device, torch_dtype = torch.bfloat16)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    EOS_TOKEN = tokenizer.eos_token # End of sequence token
    tokenizer.pad_token = tokenizer.eos_token

    # GENERATE OUTPUT
    question = """
    You are an assistant for question-answering tasks. Answering and explaining the questions appropriately.
    
    ### Question:
    In a patient with fresh blow out fracture of the orbit, best immediate management is

    ### Answer:
"""
 
    print(generate_output(model, tokenizer, question))
    

if __name__ == "__main__":
    main()
