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

def cov(answer: str, question: str):
    
    plan_verification_msgs = [
        {"role": "system", "content": 'Generate different questions for each single fact in the answer.'},
        {"role": "user", "content": f"Question: {question}\n\nAnswer: {answer}"},
    ]

    input_tokens = tokenizer.apply_chat_template(
        plan_verification_msgs,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device) #type: ignore

    verification_question_tokens = model.generate(input_tokens, max_new_tokens=max_tokens, eos_token_id=terminators, do_sample=True, top_p=0.9) #type: ignore
    verifcation_question = tokenizer.decode(verification_question_tokens[0, input_tokens.shape[-1]:], skip_special_tokens=True)

    print(verifcation_question)

    excute_verification_msgs = [
        {"role": "system", "content": "Answer the question directly. Do not provide any unnecessary information."},
        {"role": "user", "content": f"Question: {verifcation_question}"},
    ]

    input_tokens = tokenizer.apply_chat_template(
        excute_verification_msgs,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device) #type: ignore

    verification = model.generate(input_tokens, max_new_tokens=max_tokens, eos_token_id=terminators, do_sample=True, top_p=0.9) #type: ignore
    verification_answer = tokenizer.decode(verification[0, input_tokens.shape[-1]:], skip_special_tokens=True)

    print(verification_answer)

    return verification_answer

cov("The Mexican American War was an armed conflict between the United States and Mexico from 1864 to 1868. It followed in the wake of the 1845 U.S annexation of Texas, which Mexico considered part of its territory in spite of its de facto secession in the 1835 Texas Revolution.", "What was the primary cause of the Mexican American War?")