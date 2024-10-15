import gradio as gr
from Self_refine.self_refine import self_refine

def generate(message, history):
    return self_refine(message)

iface = gr.ChatInterface(
    generate,
    type="messages",
    examples=[{"text": "Why is Canberra the capital of Australia?"}, {"text": "How COVID-19 spreads?"}, {"text": "What can lacking vitamin D cause?"}],
)

iface.launch(share=True)