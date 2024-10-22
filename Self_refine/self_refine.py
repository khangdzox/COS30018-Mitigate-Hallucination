import transformers, peft, warnings
from ..hallucination_detection.detection import selfcheckgpt

fred = "\x1b[38;5;1m"
fyellow = "\x1b[38;5;3m"
fgreen = "\x1b[38;5;2m"
reset = "\x1b[0m"

warnings.simplefilter(action="ignore", category=FutureWarning)

def self_refine(
    question: str,
    model: transformers.PreTrainedModel | peft.peft_model.PeftModel,
    tokenizer: transformers.PreTrainedTokenizer,
    terminators: list[int],
    max_iterations = 4,
    max_new_tokens = 100) -> str:

    def model_generate(input_msgs: list[dict[str, str]], continuing=False):

        input_tokens = tokenizer.apply_chat_template(
            input_msgs,
            add_generation_prompt=not continuing,
            continue_final_message=continuing,
            return_tensors="pt",
        ).to(model.device) #type: ignore

        output_tokens = model.generate(input_tokens, max_new_tokens=max_new_tokens, eos_token_id=terminators, do_sample=True, top_p=0.9) #type: ignore
        return tokenizer.decode(output_tokens[0, input_tokens.shape[-1]:], skip_special_tokens=True)

    input_msgs = [
        {"role": "system", "content": "Answer the question directly. Do not provide any unnecessary information."},
        {"role": "user", "content": question},
    ]

    generation_str = tokenizer.apply_chat_template(input_msgs, tokenize=False)

    answer = model_generate(input_msgs)

    print(f"\n{fred}Question{reset}: {question}\n{fgreen}Initial answer{reset}: {answer}")

    for _ in range(max_iterations):

        # Get the feedback for the answer
        feedback_input_msgs = [
            {"role": "system", "content": "Provide short, concise, actionable and specific feedbacks to ensure the correctness of your answer. Do not answer the question or provide example."},
            {"role": "user", "content": f"Question: {question}\n\nAnswer: {answer}"},
        ]

        feedback = model_generate(feedback_input_msgs)

        print(f"{fyellow}Feedback{reset}: {feedback}")

        # Refine the answer based on the feedback
        refined_input_msgs = [
            {"role": "system", "content": "Provide a new short, concise, refined answer using the feedback. Do not provide additional statements."},
            {"role": "user", "content": f"Question: {question}\n\nAnswer: {answer}\n\nFeedback: {feedback}"},
        ]

        refined_answer = model_generate(refined_input_msgs)

        print(f"{fgreen}Refined answer{reset}: {refined_answer}")

        # If the refinement is not hallucinated then break
        if selfcheckgpt(generation_str, refined_answer, model, tokenizer, terminators, 5):
            break

        answer = refined_answer

    return refined_answer
