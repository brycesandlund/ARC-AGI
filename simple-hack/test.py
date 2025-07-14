from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model
import wandb
import torch
from functions import generate_math_problems, math_reward_func, test_inference

model_name = "Qwen/Qwen3-1.7B"
dataset_size = 400
experiment_name = "math-24-game-2"









def train():
    """
    Train the model using GRPO with math problems generated from generate_problem function.
    """
    # Initialize wandb
    wandb.init(
        project="math-grpo-training",
        name=experiment_name
    )
    
    # Load tokenizer first to format prompts
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create separate train and eval datasets
    full_dataset = Dataset.from_generator(
        generate_math_problems, gen_kwargs={"tokenizer": tokenizer, "dataset_size": dataset_size}
    )
    dataset_size_used = min(len(full_dataset), dataset_size)

    # Split into train/eval with fixed eval size of 30
    eval_size = 30
    train_size = dataset_size_used - eval_size
    train_dataset = full_dataset.select(range(train_size))
    eval_dataset = full_dataset.select(range(train_size, dataset_size_used))

    print(f"Created dataset with {len(train_dataset)} training examples and {len(eval_dataset)} evaluation examples")
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Configure GRPO training arguments
    grpo_config = GRPOConfig(
        output_dir=f"./{experiment_name}",
        run_name=experiment_name,
        weight_decay=0.01,
        max_grad_norm=1.0,
        learning_rate=5e-5,  # Slightly lower learning rate for more stable training
        lr_scheduler_type="cosine",
        per_device_train_batch_size=8,  # Increased batch size for A100. Reduce if you encounter OOM errors.
        num_generations=8,
        gradient_accumulation_steps=1,  # Effective batch size of 8
        num_train_epochs=1,
        max_completion_length=1500,
        temperature=0.7,
        logging_steps=1,
        report_to="wandb",
        gradient_checkpointing=True,        # Ran out of memory on an A100 without this.
        warmup_steps=5,  # ~10% of total steps
        bf16=True,
    )

    # Initialize trainer with math reward function
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        processing_class=tokenizer,
        reward_funcs=[math_reward_func],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Running initial evaluation before training...")
    initial_eval_results = trainer.evaluate()
    print(f"Initial eval results: {initial_eval_results}")

    print("Starting GRPO training...")
    trainer.train()

    print("Pushing model to Hugging Face Hub...")
    trainer.model.push_to_hub(f"bcsandlund/{experiment_name}")

    print("Training completed! Running final evaluation...")
    final_eval_results = trainer.evaluate()
    print(f"Final eval results: {final_eval_results}")

    wandb.finish()

def main():
    train()
    # test_inference(model_name)

if __name__ == "__main__":
    main()