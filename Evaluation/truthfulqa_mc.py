import transformers, datasets, torch

def load_model() -> transformers.PreTrainedModel:
    return transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto", torch_dtype=torch.bfloat16)

model = load_model()

dataset = datasets.load_dataset("truthfulqa/truthful_qa", "multiple_choice")
# DatasetDict({
#     validation: Dataset({
#         features: ['question', 'mc1_targets', 'mc2_targets'],
#         num_rows: 817
#     })
# })


if not isinstance(dataset, datasets.DatasetDict):
    raise ValueError("Something gone wrong! TruthfulQA dataset should be of type DatasetDict")

dataset = dataset["validation"]
# Dataset({
#     features: ['question', 'mc1_targets', 'mc2_targets'],
#     num_rows: 817
# })


# to be continued...

