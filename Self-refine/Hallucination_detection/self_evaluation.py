from calendar import c
import transformers, torch
from .utils import compute_transition_scores_from_string

# model_id = "google/gemma-2-2b-it"
# tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
# model = transformers.AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)

# terminators = [
#     tokenizer.eos_token_id,
#     tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]

# question = "How many letter R are there in strawberry?"

# num_samples = 10




# # Step 1: Generate a response A using a question prompt

# messages = [
#     {"role": "system", "content": "Answer the question directly. Do not provide any unnecessary information."},
#     {"role": "user", "content": question},
# ]

# input_ids = tokenizer.apply_chat_template(
#     messages,
#     add_generation_prompt=True,
#     return_tensors="pt",
# ).to(model.device) # type: ignore

# output_ids = model.generate(
#     input_ids, # type: ignore
#     max_new_tokens=100,
#     do_sample=True,
#     top_p=0.9,
#     eos_token_id=terminators,
# )

# answer = tokenizer.decode(output_ids[0, input_ids.shape[-1]:], skip_special_tokens=True).strip()




# # Step 2: Sample 10 more responses Rs using the same question prompt

# output_ids = model.generate(
#     input_ids, # type: ignore
#     max_new_tokens=100,
#     do_sample=True,
#     top_p=0.9,
#     eos_token_id=terminators,
#     num_return_sequences=num_samples,
# )

# responses = tokenizer.batch_decode(output_ids[:, input_ids.shape[-1]:], skip_special_tokens=True)




# # Step 3: Ask the model using the following format:
# # ```
# # Question: {question}
# # Here are some brainstormed ideas: {Rs}
# # Possible answer: {A}
# # Is the possible answer:
# # A. True
# # B. False
# # The possible answer is:
# # ```

# messages = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
# Only answer True or False<|start_header_id|>user<|end_header_id|>
# Question: How many letter R are there in strawberry?
# Here are some brainstormed ideas: There a three Rs in strawberry
# Strawberry has 3 Rs
# The word'strawberry' contains 2 letter Rs
# Possible answer: there are 3 Rs
# Is the possible answer:
# A. True
# B. False<|eot_id|><|start_header_id|>assistant<|end_header_id|>
# The possible answer is:"""

# input_ids = tokenizer(messages, return_tensors="pt").to(model.device)

# output_ids = model.generate(
#     input_ids,
#     max_new_tokens=10,
#     do_sample=False,
#     eos_token_id=terminators,
# )

# output = tokenizer.decode(output_ids[0, input_ids.shape[-1]:], skip_special_tokens=True)

# # Return the output
# if "A" in output or "True" in output:
#     print(True)
# else:
#     print(False)




# The function

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

    # Ask the model to evaluate the answer
    messages = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Only answer True or False<|start_header_id|>user<|end_header_id|>

Question: {question}
Here are some brainstormed ideas:
{"\n".join(responses)}
Possible answer: {answer}
Is the possible answer:
A. True
B. False<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The possible answer is: """

    validate_input_ids = tokenizer.encode(
        messages,
        return_tensors="pt"
    ).to(model.device) # type: ignore

    # Calculate the log probabilities of the model answer A. True and B. False
    token_a_true = tokenizer.encode("A. True", return_tensors="pt").to(model.device) # type: ignore
    token_b_false = tokenizer.encode("B. False", return_tensors="pt").to(model.device) # type: ignore

    token_output_a_true = torch.cat([validate_input_ids, token_a_true], dim=1), # type: ignore
    token_output_b_false = torch.cat([validate_input_ids, token_b_false], dim=1), # type: ignore

    true_probs = compute_transition_scores_from_string(model, tokenizer, terminators, token_output_a_true, start_idx=validate_input_ids.shape[-1])
    false_probs = compute_transition_scores_from_string(model, tokenizer, terminators, token_output_b_false, start_idx=validate_input_ids.shape[-1])

    # Calculate the sum of the probabilities
    sum_true_prob = true_probs.sum().cpu().numpy().item()
    sum_false_prob = false_probs.sum().cpu().numpy().item()

    # Return the output: True if probability of A. True is greater than B. False, False otherwise
    return sum_true_prob > sum_false_prob
