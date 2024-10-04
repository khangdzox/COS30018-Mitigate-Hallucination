
import transformers, torch, datasets, evaluate, tqdm, pandas as pd

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

def load_model() -> transformers.PreTrainedModel:
    quantization_config = transformers.BitsAndBytesConfig(load_in_4bit=True)
    # return transformers.AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
    return transformers.AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=quantization_config)

def load_tokenizer() -> transformers.PreTrainedTokenizer:
    return transformers.AutoTokenizer.from_pretrained(model_id) # type: ignore

def load_terminators(tokenizer: transformers.PreTrainedTokenizer) -> list[int]:
    return [
        tokenizer.eos_token_id, # type: ignore
        # tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

print("Loading model...")
model = load_model()

print("Loading tokenizer and terminators...")
tokenizer = load_tokenizer()
terminators = load_terminators(tokenizer)

# Which magazine was started first Arthur's Magazine or First for Women?
answer = "First for Women was started first."

answer_tokens = tokenizer.encode(answer, return_tensors="pt").to(model.device) # type: ignore

keywords_input_template = [
    {"role": "system", "content": "Identify all the important keyphrases from the provided sentence and return a comma separated list."},
    {"role": "user", "content": "John Russell Reynolds was an English physician and neurologist who made significant contributions to the field of neurology."},
    {"role": "assistant", "content": "John Russell Reynolds, English, physician, neurologist, neurology"},
    {"role": "user", "content": "He was born in London in 1820 and studied medicine at the University of London."},
    {"role": "assistant", "content": "London, 1820, medicine, University of London"},
    {"role": "user", "content": "After college, he worked as a lawyer for the PGA Tour, eventually becoming the Tour's Deputy Commissioner in 1989."},
    {"role": "assistant", "content": "college, lawyer, PGA Tour, Deputy Commissioner, 1989"},
    {"role": "user", "content": "Nature Discovery"},
    {"role": "assistant", "content": "Nature Discovery"},
    {"role": "user", "content": answer},
]

keywords_input_tokens = tokenizer.apply_chat_template(
    keywords_input_template,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device) # type: ignore

keywords_output_tokens = model.generate(
    keywords_input_tokens, # type: ignore
    max_new_tokens=answer_tokens.shape[-1] * 2, # double the number of tokens in the answer
    eos_token_id=terminators,
).cpu() # type: ignore

keywords = tokenizer.decode(keywords_output_tokens[0, keywords_input_tokens.shape[-1]:], skip_special_tokens=True).split(",")
keywords = [keyword.strip() for keyword in keywords]

print(keywords)
