import transformers, peft, warnings

colored = False

fred = "\x1b[38;5;1m" if colored else ""
fyellow = "\x1b[38;5;3m" if colored else ""
fgreen = "\x1b[38;5;2m" if colored else ""
reset = "\x1b[0m" if colored else ""

warnings.simplefilter(action="ignore", category=FutureWarning)

def cov(
    question: str,
    model: transformers.PreTrainedModel | peft.peft_model.PeftModel,
    tokenizer: transformers.PreTrainedTokenizer,
    terminators: list[int],
    max_new_tokens = 100) -> str:

    def model_generate(input_msgs: list[dict[str, str]], continuing=False):

        input_tokens = tokenizer.apply_chat_template(
            input_msgs,
            add_generation_prompt=not continuing,
            continue_final_message=continuing,
            return_tensors="pt",
        ).to(model.device) #type: ignore

        output_tokens = model.generate(input_tokens, max_new_tokens=max_new_tokens, eos_token_id=terminators) #type: ignore
        return tokenizer.decode(output_tokens[0, input_tokens.shape[-1]:], skip_special_tokens=True)

    input_msgs = [
        {"role": "system", "content": "You are a virtual assistant. Answer the question directly. Do not provide any unnecessary information. If there is no single correct answer, say \"I have no comment\"."},
        {"role": "user", "content": question},
    ]

    answer = model_generate(input_msgs)

    print(f"\n{fred}Question{reset}: {question}\n{fgreen}Initial answer{reset}: {answer}")

    verify_questions_msgs = [
        {"role": "system", "content": "Generate different questions for each single fact in the answer. Questions must end with a question mark and be separated by semicolons. Do not provide introductory statement."},
        {"role": "user", "content": f"Question: {question}\n\nAnswer: {answer}"},
    ]

    verify_questions = model_generate(verify_questions_msgs)

    print(f"{fyellow}Verification questions{reset}: {verify_questions}")

    verify_answers_msgs = [
        {"role": "system", "content": "Provide a full answer for each question. Do not repeat the questions. Answers must be separated by semicolons. Do not provide any unnecessary information. Do not provide introductory statement."},
        {"role": "user", "content": verify_questions},
    ]

    verify_answers = model_generate(verify_answers_msgs)

    print(f"{fyellow}Verification answer{reset}: {verify_answers}")

    questions_list = verify_questions.split("; ")
    questions_list = [q + "?" if q[-1] != "?" else q for q in questions_list]

    answers_list = verify_answers.split("; ")
    answers_list = [a + "." if a[-1] != "." else a for a in answers_list]

    question_answer_pairs = "\n\n".join([" ".join([q, a]) for q, a in zip(questions_list, answers_list)])

    print(f"{fyellow}Verification QA pairs{reset}: {question_answer_pairs}")

    revise_msgs = [
        {"role": "system", "content": "Update the answer based on the external sources. Do not provide any unnecessary information. Do not provide introductory statement. If there is no single correct answer, say \"I have no comment\"."},
        {"role": "user", "content": f"Question: {question}\n\nAnswer: {answer}\n\nExternal sources: {question_answer_pairs}"},
    ]

    revise = model_generate(revise_msgs)

    print(f"{fgreen}Revised answer{reset}: {revise}")

    return revise
