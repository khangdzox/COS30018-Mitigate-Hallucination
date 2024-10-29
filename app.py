import transformers, torch, peft, gc, gradio as gr
from .self_refine.self_refine import self_refine

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

def change_model(current_model: gr.State, current_selection, new_selection):
    if current_selection != new_selection:
        current_model.value = None
        gc.collect()
        torch.cuda.empty_cache()

        if new_selection == 'Base model':
            return load_model(model_id), new_selection

        return load_model(model_id, lora_ver), new_selection

def changee_model_selection(current_selection, new_selection):
    return gr.Button(interactive = new_selection != current_selection)

def generate(message, chat_history, model, method):
    if method == "No method":
        template = [{"role": "user", "content": message}]
        input_tokens = tokenizer.apply_chat_template(template, return_tensors="pt").to(model.device) # type: ignore
        output_tokens = model.generate(input_tokens, max_new_tokens=200, eos_token_id=terminators, top_k=1) #type: ignore
        response = tokenizer.decode(output_tokens[0, input_tokens.shape[-1]:], skip_special_tokens=True)
    else:
        response = self_refine(message, model, tokenizer, terminators, max_new_tokens=200)

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": response})

    return "", chat_history

chat = gr.ChatInterface(
    generate,
    type="messages",
    examples=[{"text": "Why is Canberra the capital of Australia?"}, {"text": "How COVID-19 spreads?"}, {"text": "What can lacking vitamin D cause?"}],
)

with gr.Blocks(fill_height=True) as app:
    model_state = gr.State(lambda: model)
    current_model_selection = gr.State("Base model")

    with gr.Row(equal_height=True):
        method = gr.Dropdown(["Self-refine", "No method"], label="Method", container=False)
        model_selection = gr.Dropdown(["Base model", "Fine-tuned model"], label="Model", container=False)
        change_model_btn = gr.Button("Change model")

    model_selection.change(
        changee_model_selection, [current_model_selection, model_selection], [change_model_btn]
    )
    change_model_btn.click(
        change_model, [model_state, current_model_selection, model_selection], [model_state, current_model_selection]
    )
    change_model_btn.interactive = False

    chatbot = gr.Chatbot(type="messages", height=200)
    msg = gr.Textbox(submit_btn=True, label="Message", placeholder="Type your message here")
    clear = gr.ClearButton(value="Clear Chat")

    msg.submit(generate, [msg, chatbot, model_state, method], [msg, chatbot])

app.launch(share=True)