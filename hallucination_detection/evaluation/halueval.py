"""
Evaluating hallucination detection methods on HaluEval dataset, QA_samples subset.
"""

import transformers, torch, datasets, evaluate, tqdm, pandas as pd
from ..detection import self_evaluation, low_confidence_generation, selfcheckgpt

save_file = "halueval_results.csv"
save_interval = 10

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

evaluated_methods = ['self_evaluation', 'low_confidence_generation', 'selfcheckgpt']

def load_model() -> transformers.PreTrainedModel:
    quantization_config = transformers.BitsAndBytesConfig(load_in_4bit=True)
    # return transformers.AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
    return transformers.AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=quantization_config)

def load_tokenizer() -> transformers.PreTrainedTokenizer:
    return transformers.AutoTokenizer.from_pretrained(model_id) # type: ignore

def load_terminators(tokenizer: transformers.PreTrainedTokenizer) -> list[int]:
    return [
        tokenizer.eos_token_id, # type: ignore
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

print("Loading model...")
model = load_model()

print("Loading tokenizer and terminators...")
tokenizer = load_tokenizer()
terminators = load_terminators(tokenizer)

print("Loading dataset...")
dataset = datasets.load_dataset("pminervini/HaluEval", "qa_samples", split="data")
# Dataset({
#     features: ['knowledge', 'question', 'answer', 'hallucination'],
#     num_rows: 10000
# })

if not isinstance(dataset, datasets.Dataset):
    raise ValueError("Something gone wrong! HaluEval dataset should be of type Dataset")

print("Loading previous results...")
try:
    results = pd.read_csv(save_file)

    for method in evaluated_methods:
        if method not in results.columns:
            results[method] = pd.NA

except FileNotFoundError:
    results = pd.DataFrame(columns=['targets', *evaluated_methods])
    results['question'] = dataset['question']
    results['targets'] = list(map(lambda x: 1 if x == "yes" else 0, dataset['hallucination']))
    results.to_csv(save_file)
finally:
    results.set_index('question', inplace=True)

for method in evaluated_methods:

    print(f"Running {method}...")

    for knowledge, question, answer, i in tqdm.tqdm(zip(dataset['knowledge'], dataset['question'], dataset['answer'], range(len(dataset))), method, total=len(dataset)):

        # Skip if the result is already known
        if results.loc[question, method] == 0 or results.loc[question, method] == 1:
            continue

        # Add a period at the end of the knowledge if it doesn't have one
        if knowledge[-1] != ".":
            knowledge += "."

        question_with_context = f"{knowledge} {question}"
        print(question_with_context, answer)

        if method == 'self_evaluation':
            predict = self_evaluation(question_with_context, answer, 5, model, tokenizer, terminators)
        elif method == 'low_confidence_generation':
            predict = low_confidence_generation(question_with_context, answer, model, tokenizer, terminators)
        elif method == 'selfcheckgpt':
            predict = selfcheckgpt(question_with_context, answer, model, tokenizer, terminators, num_samples=5)

        results.loc[question, method] = int(predict)

        if i % save_interval == 0:
            results.to_csv(save_file)

results.to_csv(save_file)

metrics = evaluate.combine(['accuracy', 'f1', 'precision', 'recall'])

scores = {}

for method in evaluated_methods:

    print(f"Computing scores for {method}...")
    scores[method] = metrics.compute(results[method], results['targets'])

print(scores)
