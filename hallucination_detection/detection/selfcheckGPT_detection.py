import numpy as np
import spacy
from selfcheckgpt.modeling_selfcheck import SelfCheckLLMPrompt
import torch
import transformers, peft

# Load the model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nlp = spacy.load("en_core_web_sm")

class CustomSelfCheckLLMPrompt(SelfCheckLLMPrompt):
    def __init__(
        self,
        model,
        tokenizer
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        self.device = model.device
        self.prompt_template = "Context: {context}\n\nSentence: {sentence}\n\nIs the sentence supported by the context above? Answer Yes or No.\n\nAnswer: "
        self.text_mapping = {'yes': 0.0, 'no': 1.0, 'n/a': 0.5}
        self.not_defined_text = set()
        print(f"SelfCheck-LLMPrompt initialized to device {self.device}")

def selfcheckgpt(
        question,
        answer,
        model: transformers.PreTrainedModel | peft.peft_model.PeftModel,
        tokenizer: transformers.PreTrainedTokenizer,
        terminators: list[int],
        num_samples=1) -> bool:

    selfcheck = CustomSelfCheckLLMPrompt(model, tokenizer)  # Initialize the hallucination detection mechanism

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
    samples_ids = model.generate(input_ids, do_sample=True, max_new_tokens=200, num_return_sequences=num_samples, eos_token_id=terminators) #type:ignore
    samples = tokenizer.batch_decode(samples_ids[:, input_ids.shape[-1]:], skip_special_tokens=True) #type:ignore

    # Split response into sentences
    sentences = [sent.text.strip() for sent in nlp(answer).sents]

    # Use SelfCheckLLMPrompt to assess hallucination scores
    sent_scores_prompt = selfcheck.predict(
        sentences=sentences,
        sampled_passages=samples)

    # Calculate average hallucination score
    hallucination_score = np.mean(sent_scores_prompt)

    # Print and return the generated text and hallucination score
    return hallucination_score > 0.5
