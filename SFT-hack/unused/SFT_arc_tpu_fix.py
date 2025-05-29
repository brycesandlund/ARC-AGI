import os
import torch
import wandb
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

def train_on_tpu(index):
    # TPU setup
    device = xm.xla_device()
    
    # Initialize wandb for this process if it's the main process
    if xm.is_master_ordinal():
        wandb.init(project="SFT_ARC_TPU")
        print("Running on TPU type: " + str(os.environ.get('TPU_ACCELERATOR_TYPE', 'unknown')))

    BASE_MODEL_NAME = "Qwen/Qwen3-0.6B-Base"

    # Load the model with bfloat16 for best TPU performance
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16
    )
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = '<|endoftext|>'
    tokenizer.eos_token_id = 151643

    # Load dataset
    dataset = load_dataset("bcsandlund/arc-agi-prompts")['train']
    
    # Optimize batch size based on TPU capacity
    # v3-8 can handle larger batch sizes than your original setting
    per_device_batch = 4  # Increase this based on memory availability

    training_args = SFTConfig(
        output_dir="./tpu_trained_model",
        run_name="tpu-sft",
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        logging_steps=10,  # Increased to reduce XLA compilation overhead
        bf16=True,  # TPUs work best with bfloat16
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=4,  # Increased for better utilization
        max_length=1000,
        num_train_epochs=4,
        save_steps=100,  # More frequent checkpoints for preemptible TPUs
        max_grad_norm=0.2,
        report_to="wandb" if xm.is_master_ordinal() else "none",  # Only report from main process
        eos_token='<|endoftext|>',
        # TPU-specific settings
        dataloader_drop_last=True,  # Important for TPU efficiency
        dataloader_num_workers=8,   # Optimize data loading
        save_total_limit=3,         # Keep only the last few checkpoints
    )

    # TPU-optimized trainer setup
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # Special TPU handling for preemptions
    try:
        trainer.train()
        # Mark step to ensure all TPU operations are complete
        xm.mark_step()
        
        # Save model on master process
        if xm.is_master_ordinal():
            trainer.save_model()
            print("Training completed successfully and model saved!")
    except Exception as e:
        # Handle preemption or other errors
        if xm.is_master_ordinal():
            print("Training interrupted: " + str(e))
            # Save checkpoint if possible
            try:
                trainer.save_model("./interrupted_checkpoint")
                print("Saved emergency checkpoint")
            except:
                print("Could not save emergency checkpoint")

if __name__ == "__main__":
    # Detect number of TPU cores
    # For v3-8, this should be 8 cores
    num_cores = os.environ.get('TPU_NUM_DEVICES', None)
    if num_cores is None:
        try:
            import torch_xla.utils.utils as xu
            num_cores = xu.getenv_as('XRT_DEVICE_MAP', str, '').count(',') + 1
        except:
            num_cores = 8  # Default for v3-8 pod
    else:
        num_cores = int(num_cores)
    
    print("Launching training on " + str(num_cores) + " TPU cores")
    
    # For single-host, multi-core TPU (v3-8)
    xmp.spawn(train_on_tpu, nprocs=num_cores)
