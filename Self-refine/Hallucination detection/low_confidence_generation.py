import transformers, torch

model_id = "google/gemma-2-2b-it"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

question = "How many letter R are there in strawberry?"




# Step 1: Generate a response A using a question prompt

answer_input_msgs = [
    {"role": "user", "content": question},
]

answer_input_tokens = tokenizer.apply_chat_template(
    answer_input_msgs,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device) # type: ignore

answer_output_tokens = model.generate(
    answer_input_tokens,
    max_new_tokens=50,
    do_sample=True,
    top_p=0.9,
    output_scores=True,
    return_dict_in_generate=True,
).cpu()

answer_only_tokens = answer_output_tokens.sequences.squeeze(0)[answer_input_tokens.shape[-1]:].cpu()

answer = tokenizer.decode(answer_only_tokens, skip_special_tokens=True).strip()

answer_transition_scores = model.compute_transition_scores(answer_output_tokens.sequences, answer_output_tokens.scores, normalize_logits=True).squeeze(0)




# Step 2: Get the keywords from the response A using a prompt (?) and the response A

keywords_input_msgs = [
    {"role": "system", "content": "Identify all the important keyphrases from the provided content and return a comma separated list."},
    {"role": "user", "content": answer},
]

keywords_input_tokens = tokenizer.apply_chat_template(
    keywords_input_msgs,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device) # type: ignore

keywords_output_tokens = model.generate(
    keywords_input_tokens,
    max_new_tokens=50,
    do_sample=True,
    top_p=0.9
).cpu()

keywords = tokenizer.decode(keywords_output_tokens[0, keywords_input_tokens.shape[-1]:], skip_special_tokens=True).strip().split(", ")

keyword_tokens = {keyword: tokenizer.encode(keyword) for keyword in keywords}




# Step 3: For each keyword, calculate the minimum of softmax token probabilities

def find_subset_index(subset, sequence):
    for i in range(len(sequence) - len(subset) + 1):
        if sequence[i:i + len(subset)] == subset:
            return i
    return None

kw_probs = {}

for kw, toks in keyword_tokens.items():
    kwidx = find_subset_index(toks, answer_only_tokens)

    if kwidx is None:
        continue

    kw_probs[kw] = torch.min(torch.exp(answer_transition_scores[kwidx:kwidx + len(toks)]))

print(kw_probs)




# Step 4: If the minimum of softmax token probabilities is less than a threshold, then the response A is low-confidence generation

if any(prob < 0.1 for prob in kw_probs.values()):
    print("Low-confidence generation detected.")
