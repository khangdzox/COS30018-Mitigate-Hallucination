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
4. IPython for Jupyter Notebook in VSCode
```bash
pip install ipython ipywidgets
```

## Hugging Face Access Token

Llama 3 is a private model, and you need to have access to it.

1. Go to the Hugging Face website and create an account.

2. Go to the Llama 3 model page and request access.

3. Once you have access, go to your user settings and create a new access token (read token is enough).

4. Copy the access token and save it in a `secret.py` file in the root directory.

```python
# secret.py
HF_TOKEN = "your_access_token"
```