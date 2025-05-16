import torch
import wandb

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


wandb.init(project="SFT_ARC")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE_MODEL_NAME = "Qwen/Qwen3-0.6B-Base"

# load the tokenizer and the model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.pad_token = '<|endoftext|>'
tokenizer.eos_token_id = 151643

from datasets import load_from_disk

dataset = load_from_disk("./GRPO-hack/filtered_dataset/r1-arcagi-successful-trajectories-8k")

training_args = SFTConfig(
    output_dir="./",
    run_name="test-sft",
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    max_length=8000,
    num_train_epochs=4,
    save_steps=400,
    max_grad_norm=0.2,
    report_to="wandb",
    eos_token='<|endoftext|>',
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

trainer.train()
