"""
Evaluating hallucination detection methods on HaluEval dataset, QA_samples subset.
"""


import transformers, torch, datasets, evaluate, tqdm
from Hallucination_detection import self_evaluation, low_confidence_generation

def load_model() -> transformers.PreTrainedModel:
    return transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto", torch_dtype=torch.bfloat16)

def load_tokenizer() -> transformers.PreTrainedTokenizer:
    return transformers.AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct") # type: ignore

def load_terminators(tokenizer: transformers.PreTrainedTokenizer) -> list[int]:
    return [
        tokenizer.eos_token_id, # type: ignore
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

model = load_model()
tokenizer = load_tokenizer()
terminators = load_terminators(tokenizer)

dataset = datasets.load_dataset("pminervini/HaluEval", "qa_samples", split="data")
# Dataset({
#     features: ['knowledge', 'question', 'answer', 'hallucination'],
#     num_rows: 10000
# })

if not isinstance(dataset, datasets.Dataset):
    raise ValueError("Something gone wrong! HaluEval dataset should be of type Dataset")

results = {
    'targets': list(map(lambda x: 1 if x == "yes" else 0, dataset['hallucination'])),
}

for method in ['self_evaluation', 'low_confidence_generation']:
    results[method] = []

    for knowledge, question, answer in tqdm.tqdm(zip(dataset['knowledge'], dataset['question'], dataset['answer']), method):
        if knowledge[-1] != ".":
            knowledge += "."

        question_with_context = f"{knowledge} {question}"

        if method == 'self_evaluation':
            predict = self_evaluation(question_with_context, answer, 10, model, tokenizer, terminators)
        elif method == 'low_confidence_generation':
            predict = low_confidence_generation(question_with_context, answer, model, tokenizer, terminators)

        results[method].append(int(predict))

metrics = evaluate.combine(['accuracy', 'f1', 'precision', 'recall'])

scores = {}

for method in ['self_evaluation', 'low_confidence_generation']:
    scores[method] = metrics.compute(results['targets'], results[method])

print(scores)
