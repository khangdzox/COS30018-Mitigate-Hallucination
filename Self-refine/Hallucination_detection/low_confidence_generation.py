from numpy import add
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
    {"role": "system", "content": "Answer the question directly. Do not provide any unnecessary information."},
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
    eos_token_id=terminators,
).cpu()

answer_only_tokens = answer_output_tokens[0, answer_input_tokens.shape[-1]:]

answer = tokenizer.decode(answer_only_tokens, skip_special_tokens=True).strip()

def compute_transition_scores_from_string(model, tokenizer, terminators, string_tokens, start_idx=0):
    """
    Manually compute the transition scores for a string by generating one token at a time.

    Args:
        model: The model to use for generation.
        tokenizer: The tokenizer to use for generation.
        terminators: A list of token IDs that indicate the end of a sequence.
        string_tokens: The input string tokens of the full generation.
        start_idx: The index of the start of the answer.

    Returns:
        The transition scores for the string.
    """

    # logits = ()
    # for i in range(start_idx, string_tokens.shape[-1] - 1):
    #     logit = model.generate(
    #         string_tokens[:, :i],
    #         max_new_tokens=1,
    #         output_scores=True,
    #         return_dict_in_generate=True,
    #         eos_token_id=terminators,
    #     )
    #     logits += logit.scores

    # do the above in batch

    generation_batch = torch.cat([
        torch.cat([
            # left pad with eos tokens
            torch.ones((string_tokens.shape[0], string_tokens.shape[1] - i), dtype = string_tokens.dtype, device=model.device) * tokenizer.eos_token_id,
            string_tokens[:, :i]
        ], dim=-1)
        for i in range(start_idx, string_tokens.shape[-1])
    ], dim=0)
    # Example:
    # tensor([[<|eos|>, <|eos|>, <|eos|>, token1, token2],
    #         [<|eos|>, <|eos|>,  token1, token2, token3],
    #         [<|eos|>,  token1,  token2, token3, token4]])

    generation_attention_mask = torch.cat([
        torch.cat([
            torch.zeros((string_tokens.shape[0], string_tokens.shape[1] - i), dtype = string_tokens.dtype, device=model.device),
            torch.ones((string_tokens.shape[0], i), dtype = string_tokens.dtype, device=model.device)
        ], dim=-1)
        for i in range(start_idx, string_tokens.shape[-1])
    ], dim=0)
    # Example:
    # tensor([[0, 0, 0, 1, 1],
    #         [0, 0, 1, 1, 1],
    #         [0, 1, 1, 1, 1]])

    generation_logits = model.generate(
        input_ids=generation_batch,
        attention_mask=generation_attention_mask,
        max_new_tokens=1,
        output_scores=True,
        return_dict_in_generate=True,
        eos_token_id=terminators,
    )
    # Example:
    # generation_logits.sequences                         vvvvvv -> this is the token we generated
    # tensor([[<|eos|>, <|eos|>, <|eos|>, token1, token2, token3],
    #         [<|eos|>, <|eos|>,  token1, token2, token3, token4],
    #         [<|eos|>,  token1,  token2, token3, token4, token5]])
    # generation_logits.scores
    # (tensor([[-84.0000, -84.0000, -88.5000,  ..., -93.0000, -93.5000, -87.0000],
    #          [-74.0000, -73.0000, -77.5000,  ..., -77.5000, -81.5000, -75.5000],
    #          [-84.0000, -82.0000, -85.5000,  ..., -84.0000, -88.5000, -81.5000]]),
    # )

    generation_scores = torch.split(generation_logits.scores[0], 1)
    # The score should be tuple of tensors, each tensor corresponding to the score of the generated token
    # (tensor([[-84.0000, -84.0000, -88.5000,  ..., -93.0000, -93.5000, -87.0000]]),
    #  tensor([[-74.0000, -73.0000, -77.5000,  ..., -77.5000, -81.5000, -75.5000]]),
    #  tensor([[-83.5000, -81.5000, -85.0000,  ..., -83.5000, -88.0000, -81.5000]])
    # )

    transition_scores = model.compute_transition_scores(string_tokens, generation_scores, normalize_logits=True)

    return transition_scores.squeeze(0)

answer_transition_scores = compute_transition_scores_from_string(model, tokenizer, terminators, answer)




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
    top_p=0.9,
    eos_token_id=terminators,
).cpu()

keywords = tokenizer.decode(keywords_output_tokens[0, keywords_input_tokens.shape[-1]:], skip_special_tokens=True).strip().split(", ")

keyword_tokens = {keyword: tokenizer.encode(keyword) for keyword in keywords}




# Step 3: For each keyword, calculate the minimum of softmax token probabilities

def find_all_subset_index(subset, sequence):
    ans = []
    for i in range(len(sequence) - len(subset) + 1):
        if sequence[i:i + len(subset)] == subset:
            ans.append(i)
    return ans

kw_probs = {}

for kw, toks in keyword_tokens.items():
    kwidxes = find_all_subset_index(toks, answer_only_tokens)

    if not kwidx:
        continue

    for i, kwidx in enumerate(kwidxes):
        kw_probs[f"{kw} {i}"] = torch.min(torch.exp(answer_transition_scores[kwidx:kwidx + len(toks)]))

print(kw_probs)




# Step 4: If the minimum of softmax token probabilities is less than a threshold, then the response A is low-confidence generation

if any(prob < 0.1 for prob in kw_probs.values()):
    print("Low-confidence generation detected.")




# The function
def low_confidence_generation(
    question: str,
    answer: str,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    terminators: list[int],
) -> bool:

    full_string_template = [
        {"role": "system", "content": "Answer the question directly. Do not provide any unnecessary information."},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    full_string_tokens = tokenizer.apply_chat_template(
        full_string_template,
        return_tensors="pt",
    ).to(model.device) # type: ignore

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

    # get the token ids for each keyword
    keyword_tokens = {keyword: tokenizer.encode(keyword.strip()) for keyword in keywords}

    # calculate the minimum of softmax token probabilities for each keyword
    kw_probs = {}

    for kw, toks in keyword_tokens.items():
        kwidxes = find_all_subset_index(toks, answer_only_tokens)

        if kwidxes is None:
            continue

        for i, kwidx in enumerate(kwidxes):
            kw_probs[f"{kw} {i}"] = torch.min(torch.exp(answer_transition_scores[kwidx:kwidx + len(toks)]))

    # if the minimum of softmax token probabilities is less than a threshold, then the response is low-confidence generation
    return not any(prob < 0.1 for prob in kw_probs.values())