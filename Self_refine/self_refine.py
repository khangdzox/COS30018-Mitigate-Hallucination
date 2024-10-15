import transformers, torch
from ..hallucination_detection.detection import selfcheckgpt

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

def load_model() -> transformers.PreTrainedModel:
    quantization_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant = True
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=quantization_config)
    return model

model = load_model()

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id) # type: transformers.PreTrainedTokenizer # type: ignore

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

def self_refine(question: str, max_iterations = 4, max_tokens = 100) -> str:

    input_msgs = [
        {"role": "system", "content": "Answer the question directly. Do not provide any unnecessary information."},
        {"role": "user", "content": question},
    ]

    input_tokens = tokenizer.apply_chat_template(
        input_msgs,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device) #type: ignore

    answer_tokens = model.generate(input_tokens, max_new_tokens=max_tokens, eos_token_id=terminators, do_sample=True, top_p=0.9) #type: ignore
    answer = tokenizer.decode(answer_tokens[0, input_tokens.shape[-1]:], skip_special_tokens=True)

    # print(f"\nInitial answer: {answer}")

    for _ in range(max_iterations):

        # Get the feedback for the answer
        feedback_input_msgs = [
            {"role": "system", "content": "Provide concise, actionable and specific feedbacks to improve the answer."},
            {"role": "user", "content": answer}
        ]

        feedback_input_tokens = tokenizer.apply_chat_template(
            feedback_input_msgs,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device) #type: ignore

        feedback_output_tokens = model.generate(feedback_input_tokens, max_new_tokens=max_tokens, eos_token_id=terminators, do_sample=True, top_p=0.9) #type: ignore
        feedback = tokenizer.decode(feedback_output_tokens[0, feedback_input_tokens.shape[-1]:], skip_special_tokens=True)

        # print(f"\nFeedback: {feedback}")

        # Refine the answer based on the feedback
        refined_input_msgs =[
            {"role": "system", "content": "Provide new concise, improved answer using the feedback."},
            {"role": "user", "content": f"Feedback: {feedback}\n\nAnswer: {answer}"}
        ]

        refined_input_tokens = tokenizer.apply_chat_template(
            refined_input_msgs,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device) #type: ignore

        refined_tokens = model.generate(refined_input_tokens, max_new_tokens=max_tokens, eos_token_id=terminators, do_sample=True, top_p=0.9) #type: ignore
        refined_answer = tokenizer.decode(refined_tokens[0, refined_input_tokens.shape[-1]:], skip_special_tokens=True)

        # print(f"\nRefined answer: {refined_answer}")

        # If the refinement is not hallucinated then break
        if selfcheckgpt(question, refined_answer, model, tokenizer, terminators, 5):
            break

        answer = refined_answer

    return refined_answer
