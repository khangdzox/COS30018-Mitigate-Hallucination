import pandas as pd, datasets

dataset_csv_path = "C:\\Users\\khang\\Downloads\\llama3_truthfulqa_gen.csv"

dataset_og = datasets.load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
assert isinstance(dataset_og, datasets.Dataset)

dataset_og = dataset_og.to_pandas()
assert isinstance(dataset_og, pd.DataFrame)

# Convert the numpy arrays to lists
dataset_og["correct_answers"] = dataset_og["correct_answers"].apply(lambda x: x.tolist())
dataset_og["incorrect_answers"] = dataset_og["incorrect_answers"].apply(lambda x: x.tolist())

dataset_csv = pd.read_csv(dataset_csv_path)
dataset_csv['correct_answers'] = dataset_og['correct_answers']
dataset_csv['incorrect_answers'] = dataset_og['incorrect_answers']
dataset_csv.to_csv("llama3_truthfulqa_gen_fix.csv", index=False)

dataset_fix = pd.read_csv("llama3_truthfulqa_gen_fix.csv")
dataset_fix['correct_answers'] = dataset_fix['correct_answers'].apply(eval)
dataset_fix['incorrect_answers'] = dataset_fix['incorrect_answers'].apply(eval)

assert dataset_fix.equals(dataset_csv), "Datasets are not equal"