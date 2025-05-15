import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


device = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE_MODEL_NAME = "Qwen/Qwen3-0.6B-Base"

# load the tokenizer and the model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
).to(device)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.pad_token = '<|endoftext|>'
tokenizer.eos_token_id = 151643

dataset = load_dataset('tttx/r1-arcagi-successful-trajectories')['train']

training_args = SFTConfig(
    output_dir="./",
    run_name="arc-sft/test-sft",
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_length=20000,
    num_train_epochs=3,
    save_steps=100,
    max_grad_norm=0.2,
    report_to="wandb",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=training_args
)

trainer.train()
