---
library_name: peft
base_model: mistralai/Mistral-7B-v0.1
datasets:
- FinGPT/fingpt-sentiment-train
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

<details><summary>See config</summary>

```yaml
base_model: mistralai/Mistral-7B-v0.1
model_type: MistralForCausalLM
tokenizer_type: LlamaTokenizergin
is_mistral_derived_model: true

load_in_8bit: false
load_in_4bit: false
strict: false

datasets:
  - path: data.jsonl
    ds_type: json
    type:
      field_instruction: instruction
      field_input: input
      field_output: output
      
      format: |-
        [INST]{input}
        {instruction} [/INST] 


dataset_prepared_path:
val_set_size: 0.05
output_dir: ./lora-out

sequence_len: 4096
sample_packing: false
eval_sample_packing: false
pad_to_sequence_len: false

adapter: lora
lora_model_dir:
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_linear: true
lora_fan_in_fan_out:

gradient_accumulation_steps: 1
micro_batch_size: 32
num_epochs: 4
optimizer: adamw_torch
lr_scheduler: cosine
learning_rate: 0.0001

bf16: auto
fp16: false
tf32: false
train_on_inputs: false
group_by_length: false

gradient_checkpointing: true
early_stopping_patience:
resume_from_checkpoint:
local_rank:
logging_steps: 1
xformers_attention:
flash_attention: true

warmup_steps: 10
save_steps:
debug:
weight_decay: 0.0
fsdp:
fsdp_config:
special_tokens:
  bos_token: "<s>"
  eos_token: "</s>"
  unk_token: "<unk>"

```

</details><br>

# Finistral-7B Financial Sentiment Analyst

This model is a fine-tuned version of [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) on the [FinGPT Sentiment](https://huggingface.co/datasets/FinGPT/fingpt-sentiment-train) dataset. It is intended to be used for sentiment analysis tasks for financial data. 


## Python Example

```python

# Downloading necessary dependencies
!pip install -qU transformers peft datasets scikit-learn seaborn matplotlib pandas tqdm bitsandbytes gradio


from huggingface_hub import login
login(token="Your_HuggingFace_Access_Token")


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Defining model identifiers
base_model = "mistralai/Mistral-7B-v0.1"
peft_model = "Ayansk11/Finistral-7B_lora"

# Loading tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token

# Loading base model with 8-bit precision
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    load_in_8bit=True,
    trust_remote_code=True,
    use_auth_token=True
)

# Loading LoRA adapter
model = PeftModel.from_pretrained(model, peft_model)
model.eval()


def predict_sentiment(text):

    # Formatting the input text into a prompt

    prompt = f"""Instruction: What is the sentiment of this news? Please choose an answer from {{negative/neutral/positive}}
    Input: {text}
    Answer: """

    # Tokenizing the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generating Answer
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20)

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extracting the answer
    answer = decoded_output.split("Answer:")[-1].strip().split()[0].lower()

    # Determining the sentiment
    if "neg" in answer:
        return "Negative"
    elif "neu" in answer:
        return "Neutral"
    elif "pos" in answer:
        return "Positive"
    else:
        return "Neutral"  


import gradio as gr

custom_css = """
body, .gradio-container {
    background-color: #0d1117 !important;
    color: #e1e1e1 !important;
}
.gradio-container .input_txt,
.gradio-container .output_txt {
    background-color: #1e272e !important;
    border-radius: 0.5rem !important;
    padding: 1rem !important;
}
.gradio-container button {
    border-radius: 0.5rem !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
}
.gradio-container button:hover {
    background-color: #00b894 !important;
}
.gradio-container button:focus {
    outline: 2px solid #0984e3 !important;
}
"""

with gr.Blocks(
    
    title="Finistral AI: The Market Sentiment Analyst",
    theme=gr.themes.Base(),  
    css=custom_css          
) as demo:

    gr.Markdown(
        """
        <div style='text-align:center;'>
          <span style='font-size:2.5rem;'>💹</span>
          <span style='font-size:2rem; font-weight:600; color:#00b894;'>
            Finistral AI
          </span><br>
          <span style='font-size:1.25rem; color:#e1e1e1;'>
            The Market Sentiment Analyst
          </span>
        </div>
        """
    )
    gr.Markdown(
        "Paste in a financial news snippet, headline, or article "
        "to see its sentiment breakdown."
    )

    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                lines=6,
                placeholder="E.g. “Tech stocks rally as Fed signals rate cut…”",
                label="Enter Financial News"
            )
            analyze_btn = gr.Button("🔍 Analyze Sentiment")
        with gr.Column(scale=1):
            output_lbl = gr.Label(
                num_top_classes=3,
                label="Top 3 Sentiment Classes"
            )

    analyze_btn.click(
        fn=predict_sentiment,
        inputs=input_text,
        outputs=output_lbl
    )

    gr.Markdown(
        "<p style='text-align:center; color:#7f8c8d; font-size:0.875rem;'>"
        "Powered by Finistral AI • Built with Gradio"
        "</p>"
    )

demo.launch(share=True)


# Output:    
# positive
# neutral
# negative
```

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
* learning_rate: 0.0001
* train_batch_size: 32
* eval_batch_size: 32
* seed: 42
* distributed_type: multi-GPU
* num_devices: 2
* total_train_batch_size: 64
* total_eval_batch_size: 64
* optimizer: Adam with betas=(0.9, 0.999) and epsilon=1e-08
* lr_scheduler_type: cosine
* lr_scheduler_warmup_steps: 10
* num_epochs: 4

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 0.0680        | 1.0   | 1140 | 0.1121          |
| 0.1337        | 2.0   | 2280 | 0.1009          |
| 0.0499        | 3.0   | 3420 | 0.1147          |
| 0.0014        | 4.0   | 4560 | 0.1599          |

### Frameworks

* PEFT 
* Transformers 
* Pytorch 
* Datasets 
* Tokenizers 