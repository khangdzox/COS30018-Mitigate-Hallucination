import numpy as np
import torch
from selfcheckgpt.modeling_selfcheck import SelfCheckLLMPrompt
from transformers import pipeline
import spacy

def check_hallucination(question, answer, model_id, num_samples=1, device=torch.device("cuda")):
    self_check = SelfCheckLLMPrompt(model=model_id, device=device)
    pipe = pipeline("text-generation", model=model_id, device_map="auto")
    samples = pipe([question] * num_samples, temperature=1.0, do_sample=True, max_new_tokens=50, return_full_text=False)
    generated_samples = [sample[0]["generated_text"] for sample in samples]

    nlp = spacy.load("en_core_web_sm")
    Response = " ".join(generated_samples)
    sentences = [
            sent.text.strip() for sent in nlp(Response).sents
        ]  # spacy sentence tokenization
    sent_scores = self_check.predict(sentences=sentences, sampled_passages=generated_samples, verbose=False)
    hallucination_score = np.mean(sent_scores)

    return hallucination_score >= 0.5

