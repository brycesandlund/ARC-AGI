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

training_args = SFTConfig(
    output_dir="./",
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
    max_length=1000,
    num_train_epochs=3,  # Reduced epochs often work better
    save_steps=400,
    max_grad_norm=0.2,
    report_to="wandb",
    eos_token='<|endoftext|>',
    optim="adamw_8bit",  # 8-bit optimizer for memory efficiency
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

trainer.train()

# Enable fast inference mode after training
FastLanguageModel.for_inference(model)

# Save the model
model.save_pretrained("arc_model_unsloth")
tokenizer.save_pretrained("arc_model_unsloth") 