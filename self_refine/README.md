# Mitigating Halucination by Self-Refine: Iterative Refinement with Self-Feedback

Description: describe the loop, stopping conditions, etc. Impact on hallucination, how can this be used to mitigate hallucination.

## Requirements

Libraries:
- transformers
- evaluate
- datasets
- torch
- scikit-learn

## Usage

How to use the code.

## Results

Results of the experiments.

# Hallucination detection techniques

## 1. Self-evaluation
[[2207.05221] Language Models (Mostly) Know What They Know (arxiv.org)](https://arxiv.org/abs/2207.05221)

In this method, the model is used to generate one possible answer A and multiple sampling of the answer (responses Rs). The model is then given the question, the sampling responses Rs, and the generated answer A. The model is then asked if the generated answer A is True or False.

## 2. Low confidence generation detection
[[2307.03987] A Stitch in Time Saves Nine: Detecting and Mitigating Hallucinations of LLMs by Validating Low-Confidence Generation (arxiv.org)](https://arxiv.org/abs/2307.03987)

In this method, the model outputs the answer and the probability of the answer. Then, the same model is used to extract keywords from the answer. Then, we calculate the probability of each keyword. If the probability of the keyword is low, then it can be a possible signal of hallucination.

## 3. SelfCheckGPT
[[2303.08896] SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models (arxiv.org)](https://arxiv.org/abs/2303.08896)

TBA