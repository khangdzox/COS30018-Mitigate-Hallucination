# COS30018-Mitigate-Hallucination
COS30018 - Intelligent Systems. Learning and applying hallucination mitigation techniques on open LLMs.

## Requirements

Developers are advised to use a virtual environment to install the required packages. Avoid using Hugging Face with Keras or TensorFlow 3+ as it may cause compatibility issues. You can create a virtual environment with `conda` or `venv`.

This project requires Python 3.12 or later.

1. PyTorch for backend framework
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
2. 🤗 Transformers for high-level APIs
```bash
pip install transformers datasets evaluate accelerate
```
3. Gradio for web interface
```bash
pip install gradio
```