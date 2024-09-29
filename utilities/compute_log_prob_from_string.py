import torch

def compute_log_prob_from_string(model, string_tokens, start_idx=0):
    """
    Manually compute the log probability of a string.

    Args:
        model: The model to use for generation.
        string_tokens: The input string tokens of the full generation.
        start_idx: The index of the start of the answer.

    Returns:
        The log probability of the string.
    """
    # sequence_length: length of full string tokens = string_tokens.shape[-1]
    # answer_length: length of the answer = sequence_length - start_idx

    # string_tokens: tensor(batch_size, sequence_length)

    # logits: tensor(batch_size, answer_length, vocab_size)
    logits: torch.Tensor = model(string_tokens, return_dict=True, num_logits_to_keep=string_tokens.shape[-1] - start_idx).logits
    logits = logits.squeeze(dim=0) # remove batch dimension, tensor(answer_length, vocab_size)
    logits = logits.log_softmax(dim=-1) # convert logits to log probabilities

    string_tokens = string_tokens.squeeze(0) # remove batch dimension, tensor(sequence_length)
    string_tokens = string_tokens[start_idx:] # remove the prefix tokens, tensor(answer_length)

    log_prob = logits[torch.arange(logits.shape[0]), string_tokens] # tensor(answer_length)

    return log_prob
