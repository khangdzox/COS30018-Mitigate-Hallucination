import warnings
warnings.filterwarnings('ignore')

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config
)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

def self_refine(prompt: str, max_iterations: int = 2, max_tokens: int = 100) -> str:

    def is_refinement_sufficient(prompt, feedback, initial, refined) -> bool:
        return refined != initial and "better" in refined.lower()

    def LLM(prompt):
        return model.generate(prompt, max_new_tokens=max_tokens, eos_token_id=terminators, do_sample=True, top_p=0.9)[0]

    input_msgs = [
        {"role": "user", "content": prompt},
    ]

    input_tokens = tokenizer.apply_chat_template(
        input_msgs,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device) #type: ignore

    answer_tokens = LLM(input_tokens)
    answer = tokenizer.decode(answer_tokens[input_tokens.shape[-1]:],skip_special_tokens=True)

    print(f"\nInitial answer: {answer}")

    for _ in range(max_iterations):

        feedback_msgs = [
            {"role": "user", "content": f"Provide and prints a few feedbacks to improve the following answer but don't provide comments about the feedbacks, just give the feedback: {answer}"}
        ]

        feedback_input_tokens = tokenizer.apply_chat_template(
            feedback_msgs,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device) #type: ignore

        feedback_tokens = LLM(feedback_input_tokens)
        feedback = tokenizer.decode(feedback_tokens[feedback_input_tokens.shape[-1]:], skip_special_tokens=True)

        print(f"\nFeedback: {feedback}")

        refined_msgs =[
            {"role": "user", "content": f"Using this feedback: {feedback} to refine and improve the answer {answer} and print the answer after the feedback but don't provide any comments."}
        ]

        refined_input_tokens = tokenizer.apply_chat_template(
            refined_msgs,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device) #type: ignore

        refined_tokens = LLM(refined_input_tokens)
        refined_answer = tokenizer.decode(refined_tokens[refined_input_tokens.shape[-1]:], skip_special_tokens=True)

        print(f"\nRefined answer: {refined_answer}")

        if is_refinement_sufficient(input_tokens, feedback, answer, refined_answer):
            break

        answer = refined_answer

    return refined_answer


refined_answer = self_refine("What is the capital of Australia?")
