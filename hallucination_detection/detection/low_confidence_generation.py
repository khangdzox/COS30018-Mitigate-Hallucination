import transformers, torch
from ...utilities import compute_log_prob_from_string, find_all_subset_index

# Step 1: Generate a response A using a question prompt

# Step 2: Get the keywords from the response A using a prompt and the response A

# Step 3: For each keyword, calculate the minimum of softmax token probabilities

# Step 4: If the minimum of softmax token probabilities is less than a threshold, then the response A is low-confidence generation

def low_confidence_generation(
    question: str,
    answer: str,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    terminators: list[int],
    prob_threshold: float = 0.1
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

    # get the token ids of the answer. [1:] is used to remove the first token which is the BOS token
    answer_tokens = tokenizer.encode(answer)[1:]

    # calculate the transition scores (log probabilities) for each token in the answer
    answer_probs = compute_log_prob_from_string(model, full_string_tokens, start_idx=full_string_tokens.shape[-1] - len(answer_tokens) + 1)

    # get the keywords from the answer
    # keywords_input_template = [
    #     {"role": "system", "content": "Identify all the important keyphrases from the provided content and return a comma separated list."},
    #     {"role": "user", "content": answer},
    # ]

    # keywords_input_tokens = tokenizer.apply_chat_template(
    #     keywords_input_template,
    #     add_generation_prompt=True,
    #     return_tensors="pt",
    # ).to(model.device) # type: ignore

    keywords_input = f"""Identify all the important keyphrases from the provided sentence and return a comma separated list.

Q: John Russell Reynolds was an English physician and neurologist who made significant contributions to the field of neurology.
A: John Russell Reynolds, English, physician, neurologist, neurology

Q: He was born in London in 1820 and studied medicine at the University of London.
A: London, 1820, medicine, University of London

Q: After college, he worked as a lawyer for the PGA Tour, eventually becoming the Tour's Deputy Commissioner in 1989.
A: college, lawyer, PGA Tour, Deputy Commissioner, 1989

Q: Nature Discovery
A: Nature Discovery

Q: {answer}
A: """

    keywords_input_tokens = tokenizer(keywords_input, return_tensors="pt").to(model.device)

    keywords_output_tokens = model.generate(
        **keywords_input_tokens, # type: ignore
        max_new_tokens=len(answer_tokens) * 2, # double the number of tokens in the answer
        eos_token_id=terminators,
    ).cpu() # type: ignore

    keywords_output = tokenizer.decode(keywords_output_tokens[0, keywords_input_tokens["input_ids"].shape[-1]:], skip_special_tokens=True) # type: ignore
    keywords_output = keywords_output[:keywords_output.find("Q:")].strip()

    keywords = keywords_output.split(", ")
    keywords = [keyword.strip() for keyword in keywords]

    print({'question': question, 'answer': answer, 'keywords': keywords})

    keyword_tokens = {}
    # get the token ids for each keyword
    for kw in keywords:
        keyword_tokens[kw] = tokenizer.encode(kw)[1:]

        # add the token ids of the keywords with space prefix
        # because the tokenizer encodes the keywords with space prefix and no space prefix differently
        keyword_tokens[f" {kw}"] = tokenizer.encode(f" {kw}")[1:]

    # calculate the minimum of softmax token probabilities for each keyword
    kw_probs = {}

    for kw, toks in keyword_tokens.items():
        kwidxes = find_all_subset_index(toks, answer_tokens)

        if not kwidxes:
            continue

        for i, kwidx in enumerate(kwidxes):
            kw_probs[f"{kw} {i}"] = torch.min(torch.exp(answer_probs[kwidx:kwidx + len(toks)]))

    print(kw_probs)

    # if any of the minimum of softmax token probabilities is less than a threshold, then the response is low-confidence generation
    return not any(prob < prob_threshold for prob in kw_probs.values())
