import warnings
warnings.filterwarnings('ignore')

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    BitsAndBytesConfig
)
import torch

model_id = "google/gemma-2-2b-it"
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    # quantization_config=quantization_config
)

streamer = TextStreamer(tokenizer) # type: ignore

input_msgs = [
    {"role": "user", "content": "What is the capital of Australia?"},
]

input_tokens = tokenizer.apply_chat_template(
    input_msgs,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device) # type: ignore

output_tokens = model.generate(**input_tokens, streamer=streamer, max_new_tokens=100, do_sample=True, top_p=0.9)
