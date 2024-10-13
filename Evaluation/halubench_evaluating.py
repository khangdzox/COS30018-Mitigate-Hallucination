import random
from textwrap import dedent
from typing import Dict, List

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from colored import Back, Fore, Style
from datasets import Dataset, load_dataset
from matplotlib.ticker import PercentFormatter
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftConfig
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    DataCollatorForLanguageModeling
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
from nltk import word_tokenize
from rouge import rouge

COLORS = ["#bae1ff", "#ffb3ba", "#ffdfba", "#ffffba", "#baffc9"]

sns.set( style = "whitegrid", palette = "muted", font_scale = 1.2 )
sns.set_palette(sns.color_palette(COLORS))

cmap = colors.LinearSegmentedColormap.from_list("custom_cmap", COLORS[:2])

MY_STYLE = {
    "figure.facecolor": "black",
    "axes.facecolor": "black",
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "text.color": "white",
    "axes.linewidth": 0.5,
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "gray",
    "grid.linestyle": "--",
    "grid.linewidth":  0.5,
    "axes.grid": True,
    "xtick.labelsize": "medium",
    "ytick.labelsize": "medium",
    "axes.titlesize": "large",
    "axes.labelsize": "large",
    "lines.color": COLORS[0],
    "patch.edgecolor": "white",
}

mpl.rcParams.update( MY_STYLE )

SEED = 42

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# %% [markdown]
# ### **Trained Model**


# %%
seed_everything(SEED)
PAD_TOKEN = "<|pad|>"
TMODEL_NAME = "COS30018-Mitigate-Hallucination/Finetuning/QLoRA/4"
NEW_MODEL = "Llama-3-8B-Project"

# %%
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
base_model = AutoModelForCausalLM.from_pretrained(model_id, device_map='cuda', quantization_config=quantization_config)

# %%
model = PeftModel.from_pretrained(base_model, TMODEL_NAME, device_map = "cuda", torch_dtype = torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
tokenizer.padding_side = "right"

model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

# %% [markdown]
# ###Google Drive

# %%
# from google.colab import drive
# drive.mount('/content/drive')

# %%
# import zipfile
# import os

# # Replace 'your_zip_file_path' with the path to your zip file in Google Drive
# zip_path = '/content/drive/MyDrive/5.zip'
# extract_path = '/content/extracted_files'

# # Create the directory if it doesn't exist
# os.makedirs(extract_path, exist_ok=True)

# # Extract the zip file
# zip_ref = zipfile.ZipFile(zip_path, 'r')
# zip_ref.extractall(extract_path)
# zip_ref.close()

# # Verify the files are extracted
# print(os.listdir(extract_path))

# %%
# seed_everything(SEED)
# PAD_TOKEN = "<|pad|>"
# TMODEL_PATH = "/content/extracted_files/5"  # Update this with your model path
# NEW_MODEL = "Llama-3-8B-Project"

# %%
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# base_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# model = PeftModel.from_pretrained(base_model, TMODEL_PATH, device_map="cuda", torch_dtype=torch.bfloat16)

# tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
# tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
# tokenizer.padding_side = "right"

# model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

# %% [markdown]
# ### **Data Preprocessing**

# %%
dataset=load_dataset("PatronusAI/HaluBench")

# %%
dataset["test"][:2]

# %%
rows = []
for i in dataset["test"]:
    if isinstance(i["answer"], list):
        correct_answers = "; ".join(i["answer"])
    else:
        correct_answers = str(i["answer"])

    rows.append(
        {
            "question": i["question"],
            "context": i['passage'],
            "correct_answers": correct_answers,
            "label": i["label"]
        }
    )

df = pd.DataFrame(rows)

# %%
df.head()

# %%
print(df.isnull().value_counts())
fail_count = df['label'].value_counts().get('FAIL', 0)

print(f"Number of 'FAIL' occurrences: {fail_count}")

# %%
def format_example(row: dict):
    prompt = dedent(
        f"""
        ### Instruction:
        {row["context"]}

        ### Input:
        {row["question"]}

        ### Response:
        {row["correct_answers"]}

        ### Evaluation:
        """
    )
    messages = (
        {
            "role": "system",
            "content": "Read the passage and evaluate if the provided answer is correct. Respond with 'PASS' if the answer is correct and 'FAIL' if the answer is incorrect.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    )
    return tokenizer.apply_chat_template(messages, tokenize=False)

# %%
df["text"] = df.apply(format_example, axis=1)

# %%
def count_tokens(row: dict)->int:
    return len(
        tokenizer(
            row["text"],
            add_special_tokens=True,
            return_attention_mask=False,
            )["input_ids"]
        )

# %%
df["token_count"] = df.apply(count_tokens, axis=1)

# %%
df.head()

# %%
print(df.text.iloc[0])

# %%
plt.hist(df.token_count, weights=np.ones(len(df.token_count)) / len(df.token_count))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.xlabel("Token count")
plt.ylabel("Percentage")
plt.title("Token count distribution")
plt.show()

# %%
upper_bound = 1000
lower_bound = 10

# %%
len(df[(df.token_count < upper_bound) & (df.token_count > lower_bound)]), len(df), len(df[(df.token_count < upper_bound)  & (df.token_count > lower_bound)]) / len(df)

# %%
total_num = 500

# %%
df = df[(df.token_count < upper_bound) & (df.token_count > lower_bound)]
df = df.sample(total_num)
df.shape

# %%
# train, temp = train_test_split(df, test_size=0.2, random_state=SEED)
# val, test = train_test_split(temp, test_size=0.2, random_state=SEED)

# %%
# len(train) / len(df), len(val) / len(df), len(test) / len(df)

# %%
# len(train), len(val), len(test)

# %%
# train_num = 1500
# val_num = 450
# test_num = 100

# %%
# train.sample(n=train_num).to_json("train.json", orient="records", lines=True)
# val.sample(n=val_num).to_json("val.json", orient="records", lines=True)
# test.sample(n=test_num).to_json("test.json", orient="records", lines=True)

# %%
# dataset = load_dataset(
#     "json",
#     data_files={
#         "train": "train.json",
#         "validation": "val.json",
#         "test": "test.json"
#         }
#     )

# %%
test = df.sample(n=total_num)

# %%
test.to_json("test.json", orient="records", lines=True)

# %%
dataset = load_dataset(
    "json",
    data_files={
        "test": "test.json"
    }
)

# %%
dataset

# %%
print(dataset["test"][0]["text"])

# %% [markdown]
# ### **Test Base Model**

# %%
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens = 128,
    return_full_text = False,
)

# %%
def create_test_prompt(data_row):
    prompt = dedent(
        f"""
        ### Instruction:
        {data_row["context"]}

        ### Input:
        {data_row["question"]}

        ### Provided answer:
        {data_row["correct_answers"]}

        ### Response:
        """
    )
    messages = (
        {
            "role": "system",
            "content": "Read the context and evaluate if the provided answer is correct. Respond with 'PASS' if the answer is correct and 'FAIL' if the answer is incorrect.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    )
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

# %%
row = dataset["test"][0]
prompt = create_test_prompt(row)
print(prompt)


# %%
row = dataset["test"][1]
prompt = create_test_prompt(row)
print(prompt)

# %%


# %%
row = dataset["test"][2]
prompt = create_test_prompt(row)
print(prompt)

# %%
def calculate_accuracy(prediction, reference):
    pred_words = set(word_tokenize(prediction))
    ref_words = set(word_tokenize(reference))

    common_words = pred_words.intersection(ref_words)

    accuracy_pred = len(common_words) / len(pred_words) if pred_words else 0
    accuracy_ref = len(common_words) / len(ref_words) if ref_words else 0

    return accuracy_pred, accuracy_ref

# %%
from rouge import Rouge
import nltk
rows = []
rouge = Rouge()
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

for row in tqdm(dataset["test"]):
    prompt = create_test_prompt(row)
    output = pipe(prompt)
    prediction = output[0]["generated_text"]
    reference = row["label"]

    meteor = meteor_score([word_tokenize(reference)], word_tokenize(prediction), alpha=0.9, beta=3, gamma=0.5)

    bleu = sentence_bleu([word_tokenize(reference)], word_tokenize(prediction))

    rouge_scores = rouge.get_scores(prediction, reference, avg=True)

    accuracy_pred, accuracy_ref = calculate_accuracy(prediction, reference)

    rows.append(
        {
            "question": row["question"],
            "context": row["context"],
            "answer": reference,
            "prediction": prediction,
            "meteor_score": meteor,
            "bleu_score": bleu,
            "rouge_1": rouge_scores['rouge-1']['f'],
            #"rouge_2": rouge_scores['rouge-2']['f'],
            "rouge_l": rouge_scores['rouge-l']['f'],
            "Accuracy_pred": accuracy_pred,
            "Accuracy_ref": accuracy_ref,
            "Token_count": row["token_count"]
        }
    )

predictions_df = pd.DataFrame(rows)

# %%
predictions_df.to_csv('QLoRA_4.csv', index=False)


