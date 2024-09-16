import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch

model_id = "google/gemma-2-2b-it"
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# quantization_config = BitsAndBytesConfig(load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    # quantization_config=quantization_config
)

streamer = TextStreamer(tokenizer) # type: ignore

input_msgs = [
    {"role": "user", "content": "How many letter R are there in strawberry?"},
]

input_tokens = tokenizer.apply_chat_template(
    input_msgs,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device) # type: ignore

# print(input_tokens)

output_tokens = model.generate(
    input_tokens,
    streamer=streamer,
    max_new_tokens=35,
    do_sample=True,
    top_p=0.9,
    output_scores=True,
    return_dict_in_generate=True,
)

# print(output_tokens)

transition_scores = model.compute_transition_scores(output_tokens.sequences, output_tokens.scores, normalize_logits=True)

generated_tokens = output_tokens.sequences[:, input_tokens.shape[-1]:]

# print(generated_tokens)

print(f"| {'Token id':>8} | {'Token str':<20} | {'Log Prob':<8} | Probability")
for token, score in zip(generated_tokens[0], transition_scores[0]):
    print(f"| {token:>8d} | {tokenizer.decode(token).strip():<20s} | {score.cpu().numpy():>8.3f} | {torch.exp(score.cpu()):>10.2%}")
