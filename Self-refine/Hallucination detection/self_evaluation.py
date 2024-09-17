import transformers, torch

model_id = "google/gemma-2-2b-it"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

question = "How many letter R are there in strawberry?"

num_samples = 20




# Step 1: Generate a response A using a question prompt

messages = [
    {"role": "user", "content": question},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device) # type: ignore

output_ids = model.generate(
    input_ids, # type: ignore
    max_new_tokens=100,
    do_sample=True,
    top_p=0.9,
    eos_token_id=terminators,
)

answer = tokenizer.decode(output_ids[0, input_ids.shape[-1]:], skip_special_tokens=True).strip()




# Step 2: Sample 20 more responses Rs using the same question prompt

output_ids = model.generate(
    input_ids, # type: ignore
    max_new_tokens=100,
    do_sample=True,
    top_p=0.9,
    eos_token_id=terminators,
    num_return_sequences=num_samples,
)

responses = tokenizer.batch_decode(output_ids[:, input_ids.shape[-1]:], skip_special_tokens=True)




# Step 3: Ask the model using the following format:
# ```
# Question: {question}
# Here are some brainstormed ideas: {Rs}
# Possible answer: {A}
# Is the possible answer:
# A. True
# B. False
# The possible answer is:
# ```

messages = [
    {"role": "user", "content": question},
    {"role": "user", "content": f"Here are some brainstormed ideas: {responses}"},
    {"role": "user", "content": f"Possible answer: {answer}"},
    {"role": "user", "content": "Is the possible answer:\nA. True\nB. False"},
    {"role": "assistant", "content": "The possible answer is:"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device) # type: ignore

output_ids = model.generate(
    input_ids, # type: ignore
    max_new_tokens=10,
    do_sample=False,
    eos_token_id=terminators,
)

output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Return the output
if "True" in output:
    print(True)
else:
    print(False)
