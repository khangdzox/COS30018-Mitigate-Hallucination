import transformers, torch, textwrap
from ...utilities import compute_log_prob_from_string

# Step 1: Generate a response A using a question prompt

# Step 2: Sample 10 more responses Rs using the same question prompt

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

# Return the output
# Hallucination if False, Non-hallucination if True

def self_evaluation(
    question: str,
    answer: str,
    num_samples: int,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    terminators: list[int],
) -> bool:

    # Sampling responses from the question
    messages = [
        {"role": "system", "content": "Answer the question directly. Do not provide any unnecessary information."},
        {"role": "user", "content": question},
    ]

    samples_input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device) # type: ignore

    samples_output_ids = model.generate(
        samples_input_ids, # type: ignore
        max_new_tokens=100,
        do_sample=True,
        top_p=0.9,
        eos_token_id=terminators,
        num_return_sequences=num_samples,
    )

    # Get the generated part of the responses
    responses = tokenizer.batch_decode(samples_output_ids[:, samples_input_ids.shape[-1]:], skip_special_tokens=True)

    # Clean the responses
    responses = [" ".join(res.split()) for res in responses]
    joined_responses = "\n".join(responses)

    # Ask the model to evaluate the answer
    messages = [
        {"role": "user", "content": textwrap.dedent(f"""\
            Question: {question}
            Here are some brainstormed ideas:
            {joined_responses}
            Possible answer: {answer}
            Is the possible answer:
            A. True
            B. False""")},
        {"role": "assistant", "content": "The possible answer is:"}
    ]

    validate_input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        continue_final_message=True,
    ).to(model.device) # type: ignore

    # Calculate the log probabilities of the model answer A. True and B. False
    token_a_true = tokenizer.encode(" A. True", return_tensors="pt").to(model.device) # type: ignore
    token_b_false = tokenizer.encode(" B. False", return_tensors="pt").to(model.device) # type: ignore

    token_output_a_true = torch.cat([validate_input_ids, token_a_true], dim=1) # type: ignore
    token_output_b_false = torch.cat([validate_input_ids, token_b_false], dim=1) # type: ignore

    true_probs = compute_log_prob_from_string(model, token_output_a_true, start_idx=validate_input_ids.shape[-1])
    false_probs = compute_log_prob_from_string(model, token_output_b_false, start_idx=validate_input_ids.shape[-1])

    # Calculate the sum of the probabilities
    sum_true_prob = true_probs.cpu().to(torch.float32).sum().item()
    sum_false_prob = false_probs.cpu().to(torch.float32).sum().item()

    # Return the output: Hallucination if False, Non-hallucination if True
    return sum_true_prob < sum_false_prob
