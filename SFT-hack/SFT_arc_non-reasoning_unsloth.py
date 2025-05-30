import unsloth
import torch
import wandb
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel

wandb.init(project="SFT_ARC")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE_MODEL_NAME = "unsloth/Qwen2.5-0.5B-bnb-4bit"  # Use Unsloth's optimized version

# Load model and tokenizer with Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_NAME,
    max_seq_length=1000,  # Your max sequence length
    dtype=None,  # Auto-detect
    load_in_4bit=True,  # Use 4-bit quantization for memory efficiency
)

def format_chatml_prompt(messages):
    """
    Format messages into ChatML format manually
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
    
    Returns:
        Formatted string in ChatML format
    """
    formatted_text = ""
    
    for message in messages:
        role = message.get('role', 'user')
        content = message.get('content', '')
        
        if role == 'user':
            formatted_text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == 'assistant':
            formatted_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        elif role == 'system':
            formatted_text += f"<|im_start|>system\n{content}<|im_end|>\n"
    
    return formatted_text

def format_dataset_for_sft(example):
    """
    Format dataset examples for SFT training
    
    Args:
        example: Dataset example containing messages or other fields
    
    Returns:
        Dictionary with 'text' field containing formatted prompt
    """
    if 'messages' in example and example['messages']:
        # Use the custom ChatML formatting function
        formatted_text = format_chatml_prompt(example['messages'])
        return {"text": formatted_text}
    
    elif 'text' in example:
        # If already has text field, use it directly
        return {"text": example['text']}
    
    else:
        # Fallback: try to create a simple format from available fields
        content = ""
        for key, value in example.items():
            if isinstance(value, str):
                content += f"{key}: {value}\n"
        
        # Wrap in basic ChatML format
        formatted_text = f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\nProcessing...<|im_end|>\n"
        return {"text": formatted_text}

# Add LoRA adapters for efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,  # 0 is optimized for Unsloth
    bias="none",  # "none" is optimized for Unsloth
    use_gradient_checkpointing=True,
    random_state=3407,
    max_seq_length=1000,
)

dataset = load_dataset("bcsandlund/arc-agi-prompts")['train']

# Apply formatting to the dataset
dataset = dataset.map(format_dataset_for_sft, remove_columns=dataset.column_names)
import pdb; pdb.set_trace()
training_args = SFTConfig(
    output_dir="./results_unsloth",
    run_name="test-sft-unsloth",
    learning_rate=2e-4,  # Slightly higher LR works well with Unsloth
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=2,  # Can increase batch size with Unsloth
    gradient_accumulation_steps=4,  # Effective batch size of 8
    max_seq_length=1000,
    num_train_epochs=3,  # Reduced epochs often work better
    save_steps=400,
    max_grad_norm=0.2,
    report_to="wandb",
    dataset_text_field="text",  # Specify the text field
    optim="adamw_8bit",  # 8-bit optimizer for memory efficiency
    hub_model_id="bcsandlund/arc-model-unsloth",
    push_to_hub=True,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

trainer.train()

# Enable fast inference mode after training
FastLanguageModel.for_inference(model)

# Save the model locally and push to Hugging Face Hub
model.save_pretrained("arc_model_unsloth")
tokenizer.save_pretrained("arc_model_unsloth")

# Push to Hugging Face Hub
model.push_to_hub("bcsandlund/arc-model-unsloth")
tokenizer.push_to_hub("bcsandlund/arc-model-unsloth") 