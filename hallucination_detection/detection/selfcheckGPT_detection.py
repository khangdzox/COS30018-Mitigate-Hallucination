import numpy as np
import spacy
from selfcheckgpt.modeling_selfcheck import SelfCheckLLMPrompt
import torch
import transformers

# Load the model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#SelfCheckLLMPrompt(model_name): Initializes the hallucination detection class (SelfCheckLLMPrompt) with the model name.
selfcheck_prompt = SelfCheckLLMPrompt(model_name, device=device)  # Initialize the hallucination detection mechanism

nlp = spacy.load("en_core_web_sm")

def selfcheckgpt(
        question,
        answer,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        terminators: list[int],
        num_samples=1) -> bool:

    # Re-construct the input prompt from the question
    messages = [
        {"role": "system", "content": "Answer the question directly. Do not provide any unnecessary information."},
        {"role": "user", "content": question},
    ]

    # Apply the chat template to the input prompt
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device) # type: ignore

    # Generate N samples with randomness (do_sample=True) for comparison
    samples_ids = model.generate(input_ids, do_sample=True, max_new_tokens=50, num_return_sequences=num_samples, eos_token_id=terminators) #type:ignore
    samples = tokenizer.batch_decode(samples_ids[:, input_ids.shape[-1]:], skip_special_tokens=True) #type:ignore

    # Split response into sentences
    #nlp(response_text).sents: Uses SpaCy to split the generated response into sentences.
    #sent.text.strip(): Extracts the text content of each sentence and removes leading/trailing whitespace.
    sentences = [sent.text.strip() for sent in nlp(answer).sents]

    # Use SelfCheckLLMPrompt to assess hallucination scores
    #selfcheck_prompt.predict(): Compares the generated sentences against the sampled responses.
    #sent_scores_prompt: Stores the hallucination scores for each sentence in the generated response.
    #sentences=sentences, sampled_passages=samples: Inputs the original response sentences and sampled responses for analysis.
    sent_scores_prompt = selfcheck_prompt.predict(
        sentences=sentences,
        sampled_passages=samples)

    # Calculate average hallucination score
    hallucination_score = np.mean(sent_scores_prompt)

    # Print and return the generated text and hallucination score
    # print("Generated Response:", answer)
    # print("Hallucination Score:", hallucination_score)

    # if hallucination_score > 0.5:
    #     print("The response is hallucinated.")
    # else:
    #     print("The response is not hallucinated.")

    return hallucination_score
