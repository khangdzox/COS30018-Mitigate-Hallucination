# Mitigating Halucination by Finetuning

What is Fine-Tuning?
Fine-tuning involves adjusting the parameters of a pre-trained large language model (LLM) to perform better on specific tasks or domains. This process helps the model gain specialized knowledge beyond its general language understanding.

Impact on hallucination:
Fine-tuning resize the model to a specific domain only so that model will have deeper knowledge about that field and reduce the hallucination.

Fine-Tuning approach:
1. Prominent Fine-Tuning\RLHF\Parameter-Efficient Fine-Tuning(PEFT):
Freeze all of the model structure and parameter then add a little of trainable dataset on layers.
2.
3.
...

## Requirements

Framework: Huggingface, PyTorch

Library: transformers, peft

Dataset: 

## Usage

How to use the code.

## Results

Results of the experiments.