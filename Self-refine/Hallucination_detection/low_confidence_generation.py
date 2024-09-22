import transformers, torch
from .utils import compute_transition_scores_from_string, find_all_subset_index

# model_id = "google/gemma-2-2b-it"
# tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
# model = transformers.AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)

# terminators = [
#     tokenizer.eos_token_id,
#     tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]

# question = "How many letter R are there in strawberry?"




# # Step 1: Generate a response A using a question prompt

# answer_input_msgs = [
#     {"role": "system", "content": "Answer the question directly. Do not provide any unnecessary information."},
#     {"role": "user", "content": question},
# ]

# answer_input_tokens = tokenizer.apply_chat_template(
#     answer_input_msgs,
#     add_generation_prompt=True,
#     return_tensors="pt",
# ).to(model.device) # type: ignore

# answer_output_tokens = model.generate(
#     answer_input_tokens,
#     max_new_tokens=50,
#     do_sample=True,
#     top_p=0.9,
#     eos_token_id=terminators,
# ).cpu()

# answer_only_tokens = answer_output_tokens[0, answer_input_tokens.shape[-1]:]

# answer = tokenizer.decode(answer_only_tokens, skip_special_tokens=True).strip()

# answer_transition_scores = compute_transition_scores_from_string(model, tokenizer, terminators, answer)




# # Step 2: Get the keywords from the response A using a prompt (?) and the response A

# keywords_input_msgs = [
#     {"role": "system", "content": "Identify all the important keyphrases from the provided content and return a comma separated list."},
#     {"role": "user", "content": answer},
# ]

# keywords_input_tokens = tokenizer.apply_chat_template(
#     keywords_input_msgs,
#     add_generation_prompt=True,
#     return_tensors="pt",
# ).to(model.device) # type: ignore

# keywords_output_tokens = model.generate(
#     keywords_input_tokens,
#     max_new_tokens=50,
#     do_sample=True,
#     top_p=0.9,
#     eos_token_id=terminators,
# ).cpu()

# keywords = tokenizer.decode(keywords_output_tokens[0, keywords_input_tokens.shape[-1]:], skip_special_tokens=True).strip().split(", ")

# keyword_tokens = {keyword: tokenizer.encode(keyword) for keyword in keywords}




# # Step 3: For each keyword, calculate the minimum of softmax token probabilities

# kw_probs = {}

# for kw, toks in keyword_tokens.items():
#     kwidxes = find_all_subset_index(toks, answer_only_tokens)

#     if not kwidx:
#         continue

#     for i, kwidx in enumerate(kwidxes):
#         kw_probs[f"{kw} {i}"] = torch.min(torch.exp(answer_transition_scores[kwidx:kwidx + len(toks)]))

# print(kw_probs)




# # Step 4: If the minimum of softmax token probabilities is less than a threshold, then the response A is low-confidence generation

# if any(prob < 0.1 for prob in kw_probs.values()):
#     print("Low-confidence generation detected.")




# The function
def low_confidence_generation(
    question: str,
    answer: str,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    terminators: list[int],
) -> bool:

    # reconstruct the full generation string
    full_string_template = [
        {"role": "system", "content": "Answer the question directly. Do not provide any unnecessary information."},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    full_string_tokens = tokenizer.apply_chat_template(
        full_string_template,
        return_tensors="pt",
    ).to(model.device) # type: ignore

    # reconstruct the question prompt
    question_template = [
        {"role": "system", "content": "Answer the question directly. Do not provide any unnecessary information."},
        {"role": "user", "content": question},
    ]

    question_tokens = tokenizer.apply_chat_template(
        question_template,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device) # type: ignore

    # calculate the transition scores (log probabilities) for each token in the answer
    answer_transition_scores = compute_transition_scores_from_string(model, tokenizer, terminators, full_string_tokens, start_idx=question_tokens.shape[-1])

    # get the keywords from the answer
    keywords_input_template = [
        {"role": "system", "content": "Identify all the important keyphrases from the provided content and return a comma separated list."},
        {"role": "user", "content": answer},
    ]

    keywords_input_tokens = tokenizer.apply_chat_template(
        keywords_input_template,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device) # type: ignore

    keywords_output_tokens = model.generate(
        keywords_input_tokens, # type: ignore
        max_new_tokens=(full_string_tokens.shape[-1] - question_tokens.shape[-1]) * 2, # double the number of tokens in the answer
        eos_token_id=terminators,
    ).cpu() # type: ignore

    keywords = tokenizer.decode(keywords_output_tokens[0, keywords_input_tokens.shape[-1]:], skip_special_tokens=True).split(",")
    keywords = [keyword.strip() for keyword in keywords]

    print({'question': question, 'answer': answer, 'keywords': keywords})

    keyword_tokens = {}
    # get the token ids for each keyword
    for kw in keywords:
        keyword_tokens[kw] = tokenizer.encode(kw)[1:]

        # add the token ids of the keywords with space prefix
        # because the tokenizer encodes the keywords with space prefix and no space prefix differently
        keyword_tokens[f" {kw}"] = tokenizer.encode(f" {kw}")[1:]

    # get the token ids of the answer
    answer_only_tokens = tokenizer.encode(answer)[1:]

    # calculate the minimum of softmax token probabilities for each keyword
    kw_probs = {}

    for kw, toks in keyword_tokens.items():
        kwidxes = find_all_subset_index(toks, answer_only_tokens)

        if not kwidxes:
            continue

        for i, kwidx in enumerate(kwidxes):
            kw_probs[f"{kw} {i}"] = torch.min(torch.exp(answer_transition_scores[kwidx:kwidx + len(toks)]))

    print(kw_probs)

    # if any of the minimum of softmax token probabilities is less than a threshold, then the response is low-confidence generation
    return not any(prob < 0.1 for prob in kw_probs.values())