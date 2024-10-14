import numpy as np
import spacy
#AutoModelForCausalLM and AutoTokenizer: These classes from transformers load the Llama model for causal language modeling and the tokenizer for processing text inputs.
from transformers import AutoModelForCausalLM, AutoTokenizer
#SelfCheckLLMPrompt: This class from selfcheckgpt.modeling_selfcheck is used to assess the hallucination score of generated text.
from selfcheckgpt.modeling_selfcheck import SelfCheckLLMPrompt
import torch
import transformers

# Load the model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
#AutoTokenizer.from_pretrained(model_name): Loads the tokenizer for processing text input (converting strings to tokens).
tokenizer = AutoTokenizer.from_pretrained(model_name)
#AutoModelForCausalLM.from_pretrained(model_name): Loads the Llama model, specifically a causal language model, which generates text.
model = AutoModelForCausalLM.from_pretrained(model_name)  # Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
def check_hallucination(
        question,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        num_samples=1,
        nlp=spacy.load("en_core_web_sm")) -> bool:
    # Generate the initial response based on the input prompt
    #tokenizer(question): Converts the input question into tokenized form (required for input to the model).
    inputs = tokenizer(question, return_tensors="pt").to(device)  # Move inputs to GPU 
    # Extract input IDs directly
    #input_ids: Extracts the tokenized input IDs from the inputs. These IDs are the numerical representation of the input text used for model inference.
    input_ids = inputs['input_ids']
    # Generate the initial response based on the input prompt
    #model.generate(input_ids): Uses the model to generate a response based on the input tokens (input_ids).
    response = model.generate(input_ids, do_sample=False, max_new_tokens=20)
    #tokenizer.decode(response[0]): Converts the generated token IDs back into human-readable text.
    #skip_special_tokens=True: Removes special tokens (like padding or end-of-sequence tokens) from the decoded text.
    response_text = tokenizer.decode(response[0], skip_special_tokens=True)
    # Generate N samples with randomness (do_sample=True) for comparison
    #samples: A list to store the generated alternative responses.
    samples = []
    #for _ in range(num_samples): A loop that generates num_samples alternative responses.
    for _ in range(num_samples):
        sample = model.generate(input_ids, do_sample=True, temperature=1.0, max_new_tokens=20)  # Removed ** unpacking
        #samples.append(): Appends each sampled response (decoded into text) to the samples list.
        #tokenizer.decode(sample[0], skip_special_tokens=True): Converts the sampled token IDs back into human-readable text.
        #skip_special_tokens=True: Removes special tokens (like padding or end-of-sequence tokens) from the decoded text.
        #samples.append(tokenizer.decode(sample[0], skip_special_tokens=True)): Appends the decoded text to the samples list.
        samples.append(tokenizer.decode(sample[0], skip_special_tokens=True))
    
    # Split response into sentences
    #nlp(response_text).sents: Uses SpaCy to split the generated response into sentences.
    #sent.text.strip(): Extracts the text content of each sentence and removes leading/trailing whitespace.
    sentences = [sent.text.strip() for sent in nlp(response_text).sents]
    
    # Use SelfCheckLLMPrompt to assess hallucination scores
    #SelfCheckLLMPrompt(model_name): Initializes the hallucination detection class (SelfCheckLLMPrompt) with the model name.
    selfcheck_prompt = SelfCheckLLMPrompt(model_name,device=device)  # Initialize the hallucination detection mechanism
    #selfcheck_prompt.predict(): Compares the generated sentences against the sampled responses.
    #sent_scores_prompt: Stores the hallucination scores for each sentence in the generated response.
    #sentences=sentences, sampled_passages=samples: Inputs the original response sentences and sampled responses for analysis.
    sent_scores_prompt = selfcheck_prompt.predict(
        sentences=sentences,
        sampled_passages=samples)
    
    # Calculate average hallucination score
    hallucination_score = np.mean(sent_scores_prompt)
    
    # Print and return the generated text and hallucination score
    print("Generated Response:", response_text)
    print("Hallucination Score:", hallucination_score)
    
    if hallucination_score > 0.5:
        print("The response is hallucinated.")
    else:
        print("The response is not hallucinated.")
    
    return response_text, hallucination_score

# Example question
question = "one plus one is three.\nAnswer:" 
response, score = check_hallucination(question, model, tokenizer)
