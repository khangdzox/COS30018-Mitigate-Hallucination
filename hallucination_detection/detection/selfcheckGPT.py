import numpy as np
import spacy
from transformers import pipeline
from selfcheckgpt.modeling_selfcheck import SelfCheckLLMPrompt

def check_hallucination(
        question,
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        num_samples=3,
        nlp = spacy.load("en_core_web_sm"))-> bool:
    
    # Load the Llama model using Hugging Face pipeline
    pipe = pipeline("text-generation", model=model, device_map="auto")
    
    # Generate the initial response based on the input prompt
    response = pipe(question, do_sample=False, max_new_tokens=20, return_full_text=False)
    response_text = response[0]["generated_text"]
    
    # Generate N samples with randomness (do_sample=True) for comparison
    samples = pipe([question] * num_samples, temperature=1.0, do_sample=True, max_new_tokens=20, return_full_text=False)
    #samples_text: Stores the generated text of each sample in a list, with one random response per sample.
    samples_text = [sample[0]["generated_text"] for sample in samples]
    
    # Load spaCy model for sentence tokenization
   #nlp(response_text).sents: Uses the spaCy model to split response_text into individual sentences.
    sentences = [sent.text.strip() for sent in nlp(response_text).sents]  # Split response into sentences
    print(sentences)
    
    # Use SelfCheckLLMPrompt to assess hallucination scores
    llm_model = model
    selfcheck_prompt = SelfCheckLLMPrompt(llm_model)
    #selfcheck_prompt.predict(...): Passes the sentences and samples_text to SelfCheckLLMPrompt, which compares the original sentence against the random samples to detect hallucination.
    sent_scores_prompt = selfcheck_prompt.predict(
        sentences=sentences, 
        sampled_passages=samples_text)
    
    # Calculate average hallucination score Computes the mean of the hallucination scores for each sentence, which gives an overall hallucination score for the generated response.
    hallucination_score = np.mean(sent_scores_prompt)
    
    # Print and return the generated text and hallucination score
    print("Generated Response:", response_text)
    print("Hallucination Score:", hallucination_score)
    
    if hallucination_score > 0.5:
        print("The response is hallucinated.")
    else:
        print("The response is not hallucinated.")
    
    return response_text, hallucination_score

# question = "one plus one is 2.\nAnswer:"
# response, score = check_hallucination(question)
