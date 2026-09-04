# fine-tuning-LLama3# Finetuning Llama-3 8B with Unsloth

This notebook demonstrates the process of finetuning a Llama-3 8B model for various tasks, leveraging Unsloth for optimized performance, especially with 4-bit quantization.

## 1. Setup and Installation

The initial step involves setting up the environment and installing `unsloth` along with other necessary libraries for GPU acceleration.

```python
%%capture
import torch
major_version, minor_version = torch.cuda.get_device_capability()
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
if major_version >= 8:
    !pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes
else:
    !pip install --no-deps xformers trl peft accelerate bitsandbytes
pass
```

## 2. Model Loading and Quantization

We load the `llama-3-8b-bnb-4bit` model using `FastLanguageModel`, utilizing 4-bit quantization for memory efficiency.

```python
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
```

## 3. LoRA Adapters Integration

LoRA (Low-Rank Adaptation) adapters are integrated to enable efficient finetuning by updating only a small subset of the model's parameters.

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)
```

## 4. Data Preparation

The Alpaca dataset is used as an example. A formatting function is defined to prepare the data with instruction-input-response prompts and an EOS token.

```python
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

from datasets import load_dataset
dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)
```

## 5. Model Training

The model is trained using `SFTTrainer` from the `trl` library with specific training arguments for batch size, learning rate, and epochs.

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = None,
        num_train_epochs=4,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer_stats = trainer.train()
```

## 6. Inference

After training, the model can be used for inference. Examples show how to generate responses to new instructions, including using a `TextStreamer` for real-time output.

```python
FastLanguageModel.for_inference(model)
inputs = tokenizer(
[
    alpaca_prompt.format(
        "List the prime numbers contained within the range.",
        "1-50",
        "",
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True)
tokenizer.batch_decode(outputs)

# Example with TextStreamer
# FastLanguageModel.for_inference(model)
# inputs = tokenizer(
# [
#     alpaca_prompt.format(
#         "Convert these binary numbers to decimal.",
#         "1010, 1101, 1111",
#         "",
#     )
# ], return_tensors = "pt").to("cuda")

# from transformers import TextStreamer
# text_streamer = TextStreamer(tokenizer)
# _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)
```

## 7. Saving and Loading Finetuned Models

The finetuned LoRA adapters can be saved locally or pushed to the Hugging Face Hub. The notebook also demonstrates how to merge the adapters with the base model into various formats (16-bit, 4-bit, GGUF).

```python
model.save_pretrained("lora_model") # Local saving
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving

# Example of loading saved model
# if False:
#     from unsloth import FastLanguageModel
#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name = "lora_model",
#         max_seq_length = max_seq_length,
#         dtype = dtype,
#         load_in_4bit = load_in_4bit,
#     )
#     FastLanguageModel.for_inference(model)

# Merging options (set 'if False' to 'if True' to enable)
# if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
# if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
# if False: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
# if False: model.save_pretrained_gguf("model", tokenizer,)
# if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
# if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
```

## Resources

For deploying the saved `GGUF` model, tools like `llama.cpp` or UI systems like `GPT4All` can be used. Refer to the documentation of these tools for integration.
