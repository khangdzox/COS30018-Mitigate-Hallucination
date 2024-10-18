import transformers, torch, peft, gc, gradio as gr
from self_refine import self_refine

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
lora_ver = "QLoRA/7"

def load_model(model_id, lora = '') -> transformers.PreTrainedModel | peft.peft_model.PeftModel:
    quantization_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant = True
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=quantization_config)

    if lora:
        model = peft.peft_model.PeftModel.from_pretrained(model, lora, device_map="cuda", torch_dtype = torch.bfloat16)

    model.eval()
    return model

model = load_model(model_id)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id) # type: transformers.PreTrainedTokenizer # type: ignore

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

def change_method(method, prev):
    global model

    if method != prev:
        del model
        gc.collect()
        torch.cuda.empty_cache()

        if method == 'Self-refine':
            model = load_model(model_id)
        elif method == 'Fine-tune':
            model = load_model(model_id, lora_ver)

    return method

def generate(message, history):
    return self_refine(message, model, tokenizer, terminators)

chat = gr.ChatInterface(
    generate,
    type="messages",
    examples=[{"text": "Why is Canberra the capital of Australia?"}, {"text": "How COVID-19 spreads?"}, {"text": "What can lacking vitamin D cause?"}],
)

with gr.Blocks(fill_height=True) as app:

    method_selection = gr.Dropdown(["Self-refine", "Fine-tune"], label="Method")
    method_selection.change(
        change_method, [method_selection, gr.State([])], [gr.State()]
    )

    chat.render()

app.launch(share=True)