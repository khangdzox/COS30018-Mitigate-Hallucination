import transformers, torch

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

    # Example:
    # Given tokens: tensor([[token1, token2, token3, token4, token5]])

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




def find_all_subset_index(subset, sequence):
    """
    Find all the start indices of the subset in the sequence.
    """
    ans = []
    for i in range(len(sequence) - len(subset) + 1):
        if sequence[i:i + len(subset)] == subset:
            ans.append(i)
    return ans