import numpy as np
import spacy
from transformers import pipeline
import torch
from selfcheckgpt.modeling_selfcheck import SelfCheckLLMPrompt

def initialize_pipeline(device_type="auto"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", device_map=device_type)
    return pipe, device

def generate_response(pipe, prompt, sample=False, num_samples=1):
    if sample:
        responses = pipe([prompt] * num_samples, temperature=1.0, do_sample=True, max_new_tokens=20, return_full_text=False)
        return [response[0]["generated_text"] for response in responses]
    else:
        response = pipe(prompt, do_sample=False, max_new_tokens=10, return_full_text=False)
        return response[0]["generated_text"]

def detect_hallucination(sentences, samples, model_name="meta-llama/Meta-Llama-3-8B-Instruct", device=torch.device("cpu")):
    self_check = SelfCheckLLMPrompt(model_name, device=device)
    sent_scores = self_check.predict(sentences=sentences, sampled_passages=samples, verbose=False)
    hallucination_score = np.mean(sent_scores)
    if hallucination_score < 0.5:
        return "0"
    else:
        return "1"
     

def process_text(nlp, text):
    return [sent.text.strip() for sent in nlp(text).sents]

def main():
    pipe, device = initialize_pipeline()
    nlp = spacy.load("en_core_web_sm")
    
    prompt = "23 divided by 2 is 11.\nAnswer:"
    response = generate_response(pipe, prompt)
    samples = generate_response(pipe, prompt, sample=True, num_samples=1)
    
    sentences = process_text(nlp, response)
    
    is_hallucination = detect_hallucination(sentences, samples, device=device)
    return is_hallucination

if __name__ == "__main__":
    print(main())
