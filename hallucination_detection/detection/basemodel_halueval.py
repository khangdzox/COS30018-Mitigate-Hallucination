import sys, os
import transformers, torch, datasets, evaluate, tqdm, pandas as pd

# Add the directory containing hallucination_detection to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../detection')))

save_file = "halueval_results.csv"
save_interval = 10

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

def load_model() -> transformers.PreTrainedModel:
    quantization_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant = True
    )
    return transformers.AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=quantization_config)

def load_tokenizer() -> transformers.PreTrainedTokenizer:
    return transformers.AutoTokenizer.from_pretrained(model_id)

def load_terminators(tokenizer: transformers.PreTrainedTokenizer) -> list[int]:
    return [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

print("Loading model...")
model = load_model()

print("Loading tokenizer and terminators...")
tokenizer = load_tokenizer()
terminators = load_terminators(tokenizer)

print("Loading dataset...")
dataset = datasets.load_dataset("pminervini/HaluEval", "qa_samples", split="data")

if not isinstance(dataset, datasets.Dataset):
    raise ValueError("Something went wrong! HaluEval dataset should be of type Dataset")

print("Loading previous results...")
try:
    results = pd.read_csv(save_file)
except FileNotFoundError:
    results = pd.DataFrame(columns=['targets', 'base_model'])
    results['question'] = dataset['question']
    results['targets'] = list(map(lambda x: 1 if x == "yes" else 0, dataset['hallucination']))
    results.to_csv(save_file)
finally:
    results.set_index('question', inplace=True)

print("Running evaluation with the base model...")

for knowledge, question, answer, i in tqdm.tqdm(zip(dataset['knowledge'], dataset['question'], dataset['answer'], range(len(dataset))), total=len(dataset)):
    
    # Skip if the result is already known
    if results.loc[question, 'base_model'] in [0, 1]:
        continue

    # Add a period at the end of the knowledge if it doesn't have one
    if knowledge[-1] != ".":
        knowledge += "."

    question_with_context = f"{knowledge} {question}"
    
    # Generate a response using the model
    inputs = tokenizer(question_with_context, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50)  # Changed to max_new_tokens
    generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Simple logic to determine if hallucination is present
    predict = 1 if generated_answer.strip() != answer.strip() else 0
    results.loc[question, 'base_model'] = predict

    if i % save_interval == 0:
        results.to_csv(save_file)

metrics = evaluate.combine(['accuracy', 'f1', 'precision', 'recall'])
scores = {}

print("Computing scores for base model...")
scores['base_model'] = metrics.compute(results['targets'], results['base_model'])

print(scores)
