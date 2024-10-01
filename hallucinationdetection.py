import numpy as np
import spacy
import torch
from secret import HF_TOKEN
from transformers import pipeline
from huggingface_hub import login
login(token=HF_TOKEN)

from selfcheckgpt.modeling_selfcheck import SelfCheckLLMPrompt
pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct",   device_map="auto")
prompt = """
1 + 1 = 2.
Answer:
"""
Response = pipe(prompt, do_sample=False, max_new_tokens=20, return_full_text=False)
N = 20
Samples = pipe(
    [prompt] * N,
    temperature=1.0,
    do_sample=True,
    max_new_tokens=20,
    return_full_text=False,
)
Response = Response[0]["generated_text"]
Samples = [sample[0]["generated_text"] for sample in Samples]
llm_model = "meta-llama/Meta-Llama-3-8B-Instruct"
selfcheck_prompt = SelfCheckLLMPrompt(llm_model)
nlp = spacy.load("en_core_web_sm")
sentences = [
    sent.text.strip() for sent in nlp(Response).sents
]  # spacy sentence tokenization
print(sentences)

sent_scores_prompt = selfcheck_prompt.predict(
    sentences=sentences,  # list of sentences
    sampled_passages=Samples,  # list of sampled passages
    verbose=True,  # whether to show a progress bar
)

print(sent_scores_prompt)
print("Hallucination Score:", np.mean(sent_scores_prompt))