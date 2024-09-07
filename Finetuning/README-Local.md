# Mitigating Halucination by Finetuning

What is Fine-Tuning?
Fine-tuning involves adjusting the parameters of a pre-trained large language model (LLM) to perform better on specific tasks or domains. This process helps the model gain specialized knowledge beyond its general language understanding.

Impact on hallucination:
Fine-tuning resize the model to a specific domain only so that model will have deeper knowledge about that field and reduce the hallucination.

Fine-Tuning approach:
1. LoRA - Freeze pre-trained layer and only train the adapter layer (trainable parameter)
2. Hyper-parameter Tuning

## Requirements

Framework: Huggingface, PyTorch

Library: transformers, peft, datasets, torch

Dataset: Medical - link: 

## Usage

Step 1: import all require libraries. 
Note: torch need to be matched with your device version.

Step 2: Load your dataset - look at LOADDING/ load the dataset. Change the data_file or data path there.

Step 3: Modify prompt and data format - look at # DATA PREPROCESSING AND TOKENIZING, change the prompt there.
Then look at tokenize_function, change the lable and input format there.

Step 4: model's interface - look at generate_output function, Change the input_text format there to match with tokenize_function format.

## Results

Results of the experiments.