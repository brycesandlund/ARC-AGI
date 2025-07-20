import argparse
import math
from typing import List, Dict, Any, Optional
import copy

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import LinearLR
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model
import wandb

from datasets import Dataset

# Import math problem generation and reward functions
from functions import generate_math_problems, math_reward_func, parse_completion

# ============================================================
#   Generalized Reinforced Policy Optimization (GRPO)
#   Minimal PyTorch implementation written from scratch so it
#   can be easily modified later. This is NOT tied to any
#   particular reward function – plug in your own.
# ============================================================

class GRPOTrainer:
    """A minimalist GRPO trainer for causal language models."""

    def __init__(
        self,
        model: torch.nn.Module,
        ref_model: torch.nn.Module,
        lr: float = 1e-5,
        clip_ratio: float = 0.2,
        kl_coef: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dr: bool = True,
        total_steps: Optional[int] = None,
        lr_schedule: bool = True,
        min_lr_ratio: float = 0.1,
    ) -> None:
        self.model = model.to(device)
        self.ref_model = ref_model.to(device)
        self.ref_model.eval()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.clip_ratio = clip_ratio
        self.kl_coef = kl_coef
        self.device = device
        self.dr = dr
        
        # Learning rate scheduler setup
        self.lr_schedule = lr_schedule
        self.scheduler = None
        if lr_schedule and total_steps is not None:
            # LinearLR: linearly decay from 1.0 to min_lr_ratio over total_steps
            self.scheduler = LinearLR(
                self.optimizer, 
                start_factor=1.0, 
                end_factor=min_lr_ratio, 
                total_iters=total_steps
            )
            print(f"Initialized linear LR scheduler: {lr:.2e} -> {lr * min_lr_ratio:.2e} over {total_steps} steps")
            print(f"Initial optimizer LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        else:
            print(f"No LR scheduling - static LR: {self.optimizer.param_groups[0]['lr']:.2e}")

    def _old_log_probs(self, logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return log-probabilities of `actions` under the policy that produced `logits`."""
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    def _pg_loss(
        self,
        new_logp: torch.Tensor,
        old_logp: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the policy gradient loss component."""
        ratio = torch.exp(new_logp - old_logp)
        if self.clip_ratio > 0:
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            pg_loss1 = -(ratio * advantages)
            pg_loss2 = -(clipped_ratio * advantages)
            pg_losses = torch.max(pg_loss1, pg_loss2)
            
            # Debug: policy gradient loss statistics before mean
            print(f"[DEBUG PG] pg_loss1: mean={pg_loss1.mean().item():.4f}, std={pg_loss1.std().item():.4f}, min={pg_loss1.min().item():.4f}, max={pg_loss1.max().item():.4f}")
            print(f"[DEBUG PG] pg_loss2: mean={pg_loss2.mean().item():.4f}, std={pg_loss2.std().item():.4f}, min={pg_loss2.min().item():.4f}, max={pg_loss2.max().item():.4f}")
            print(f"[DEBUG PG] final_pg_losses: mean={pg_losses.mean().item():.4f}, std={pg_losses.std().item():.4f}, min={pg_losses.min().item():.4f}, max={pg_losses.max().item():.4f}")
            
            # Count clipping statistics
            clipped_mask = (ratio < 1.0 - self.clip_ratio) | (ratio > 1.0 + self.clip_ratio)
            clipped_fraction = clipped_mask.float().mean().item()
            print(f"[DEBUG PG] clipped_fraction: {clipped_fraction:.3f}")
            
            return pg_losses.mean()
        else:
            return -(ratio * advantages).mean()

    def _kl_loss(
        self,
        new_logp: torch.Tensor,
        ref_logp: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the KL divergence loss against the reference model."""
        log_ratio_ref = ref_logp - new_logp
        ratio_ref = torch.exp(log_ratio_ref)
        kl_losses = ratio_ref - log_ratio_ref - 1
        
        # Debug: KL loss statistics
        print(f"[DEBUG KL] log_ratio_ref: mean={log_ratio_ref.mean().item():.4f}, std={log_ratio_ref.std().item():.4f}, min={log_ratio_ref.min().item():.4f}, max={log_ratio_ref.max().item():.4f}")
        print(f"[DEBUG KL] kl_losses: mean={kl_losses.mean().item():.4f}, std={kl_losses.std().item():.4f}, min={kl_losses.min().item():.4f}, max={kl_losses.max().item():.4f}")
        
        return kl_losses.mean()

    def step(
        self,
        input_ids: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        prompt_length: int,
        pad_token_id: int,
        old_logp: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Run a single optimisation step on a batch.

        Parameters
        ----------
        input_ids : (B, T) full sequence tokens (prompt + generated)
        actions    : (B, G) generated tokens only
        rewards    : (B,) scalar reward for each sequence
        prompt_length : int, length of the prompt (same for all in batch)
        pad_token_id : int, token ID used for padding
        old_logp   : (B, G) old log probabilities for generated tokens only
        """
        input_ids = input_ids.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)

        # Forward pass
        outputs = self.model(input_ids)
        
        # Extract logits only for positions predicting generated tokens
        # We want logits[prompt_length-1:] to predict actions (generated tokens)
        logits = outputs.logits[:, prompt_length-1:-1, :]  # positions predicting generated tokens
        target_actions = actions  # actions are already just the generated tokens

        # Compute reference log-probs with LoRA adapters disabled to avoid parameter drift
        # If the model is a PEFT/LoRA model, temporarily turn off the adapters; otherwise
        # fall back to the separate frozen reference model.
        with torch.no_grad():
            if hasattr(self.model, "disable_adapter"):
                # LoRA or other PEFT model – disable adapters for a clean reference policy
                with self.model.disable_adapter():  # type: ignore[attr-defined]
                    ref_outputs = self.model(input_ids)
            else:
                # Full-fine-tune setting – use the dedicated frozen reference model
                ref_outputs = self.ref_model(input_ids)

            ref_logits = ref_outputs.logits[:, prompt_length-1:-1, :]
            ref_logp = self._old_log_probs(ref_logits, target_actions)

        # Compute old log-probabilities (detach from graph)
        if old_logp is None:
            old_logp = self._old_log_probs(logits.detach(), target_actions)
        else:
            old_logp = old_logp.to(self.device)

        # Compute advantages with optional sequence length normalization and standardization
        if self.dr:
            # Skip length normalization and std normalization when dr=True
            advantages = rewards.unsqueeze(-1).expand_as(old_logp)
            advantages = advantages - advantages.mean()  # Only center, don't normalize std
            advantages = advantages / 1000 # scale down advantages according to approximate sequence length
        else:
            # Apply full normalization (length + std) when dr=False
            seq_lengths = (target_actions != pad_token_id).sum(dim=1).float().clamp(min=1.0)  # Use actual pad token
            normalized_rewards = rewards / seq_lengths
            advantages = normalized_rewards.unsqueeze(-1).expand_as(old_logp)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # New log-probabilities for gradient flow
        new_logp = self._old_log_probs(logits, target_actions)

        # # Debug logging for log-probs
        # print("[DEBUG] old_logp: mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}".format(
        #     old_logp.mean().item(), old_logp.std().item(), old_logp.min().item(), old_logp.max().item()))
        # print("[DEBUG] new_logp: mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}".format(
        #     new_logp.mean().item(), new_logp.std().item(), new_logp.min().item(), new_logp.max().item()))
        # print("[DEBUG] ref_logp: mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}".format(
        #     ref_logp.mean().item(), ref_logp.std().item(), ref_logp.min().item(), ref_logp.max().item()))

        # Compute loss components
        pg_loss = self._pg_loss(new_logp, old_logp, advantages)
        kl_loss = self._kl_loss(new_logp, ref_logp)
        loss = pg_loss + self.kl_coef * kl_loss

        # Optimise
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "pg_loss": pg_loss.item(),
            "kl_loss": kl_loss.item(),
            "lr": self.optimizer.param_groups[0]['lr'],
        }

    def _run_evaluation(
        self,
        eval_dataset: Any,
        tokenizer: Any,
        max_new_tokens: int,
        eval_type: str,
        episode: int,
        use_wandb: bool = False,
    ) -> Dict[str, Any]:
        """Run evaluation and log results.
        
        Parameters
        ----------
        eval_dataset : evaluation dataset
        tokenizer : tokenizer for the model
        max_new_tokens : maximum new tokens to generate
        eval_type : "initial" or "final" for logging purposes
        episode : current episode number for wandb logging
        use_wandb : whether to log to wandb
        
        Returns
        -------
        Dict with evaluation metrics
        """
        self.model.eval()

        print(f"\nRunning {eval_type} evaluation...")
        metrics = evaluate_model(self.model, tokenizer, eval_dataset, max_new_tokens)
        print(f"{eval_type.capitalize()} metrics: {metrics}")

        if use_wandb:
            wandb.log({
                f"{eval_type}_eval/success_rate": metrics.get("eval_success_rate", 0),
                f"{eval_type}_eval/reward_mean": metrics.get("eval_reward_mean", 0),
                "episode": episode
            })
        
        return metrics

    def train(
        self,
        tokenizer: Any,
        steps: int,
        epochs_per_batch: int,
        batch_size: int,
        max_new_tokens: int,
        eval_dataset: Optional[Any] = None,
        use_wandb: bool = False,
        kl_threshold: float = 0.02,
    ) -> Dict[str, Any]:
        """Train the model using GRPO.
        
        Parameters
        ----------
        tokenizer : tokenizer for the model
        steps : total number of optimization steps
        epochs_per_batch : number of optimization steps per batch
        batch_size : batch size for training
        max_new_tokens : maximum new tokens to generate
        eval_dataset : optional dataset for evaluation
        use_wandb : whether to log to wandb
        kl_threshold : KL divergence threshold for early stopping
        
        Returns
        -------
        Dict with training statistics
        """
        total_steps = 0
        training_rewards = []
        
        # Initial evaluation if eval dataset provided
        if eval_dataset is not None:
            initial_metrics = self._run_evaluation(eval_dataset, tokenizer, max_new_tokens, "initial", 0, use_wandb)

        print("Starting GRPO training...")
        # Set the model back to train mode for training
        self.model.train()
        
        # Main training loop
        for episode in range(1, steps // epochs_per_batch + 1):
            # Sample once (expensive)
            batch = sample_math_batch(self.model, tokenizer, batch_size=batch_size, max_new_tokens=max_new_tokens)
            input_ids, actions, rewards, prompt_length, pad_token_id = batch
            
            # Track batch rewards
            batch_reward_mean = rewards.mean().item()
            batch_reward_max = rewards.max().item()
            batch_success_rate = (rewards > 0).float().mean().item()
            training_rewards.extend(rewards.tolist())
            
            # Compute old_logp ONCE from current policy
            with torch.no_grad():
                outputs = self.model(input_ids.to(self.device))
                # Extract logits only for positions predicting generated tokens
                logits = outputs.logits[:, prompt_length-1:-1, :]
                target_actions = actions.to(self.device)  # actions are already just generated tokens
                old_logp = self._old_log_probs(logits, target_actions)
            
            # Take multiple optimization steps using same old_logp
            episode_losses = []
            episode_kls = []
            
            for epoch in range(epochs_per_batch):
                total_steps += 1
                metrics = self.step(input_ids, actions, rewards, prompt_length, pad_token_id, old_logp)
                episode_losses.append(metrics['loss'])
                episode_kls.append(metrics['kl_loss'])
                
                # Step the learning rate scheduler
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Log training metrics
                if use_wandb:
                    wandb.log({
                        "train/loss": metrics['loss'],
                        "train/pg_loss": metrics['pg_loss'],
                        "train/kl_divergence": metrics['kl_loss'],
                        "train/learning_rate": metrics['lr'],
                        "train/batch_reward_mean": batch_reward_mean,
                        "train/batch_reward_max": batch_reward_max,
                        "train/batch_success_rate": batch_success_rate,
                        "episode": episode,
                        "step": total_steps
                    })
                
                print(
                    f"Episode {episode:04d}, Epoch {epoch+1:02d}/{epochs_per_batch} | "
                    f"Step: {total_steps:05d} | "
                    f"loss: {metrics['loss']:.4f} | "
                    f"kl: {metrics['kl_loss']:.4f} | "
                    f"lr: {metrics['lr']:.2e} | "
                    f"reward: {batch_reward_mean:.3f} | "
                    f"success: {batch_success_rate:.1%}"
                )
                
                # Early stopping if KL divergence gets too high
                if metrics['kl_loss'] > kl_threshold:
                    print(f"  Early stopping due to high KL divergence: {metrics['kl_loss']:.4f}")
                    break

        # Final evaluation if eval dataset provided
        if eval_dataset is not None:
            final_metrics = self._run_evaluation(eval_dataset, tokenizer, max_new_tokens, "final", episode, use_wandb)
                
            # Log training summary
            if use_wandb and wandb.run is not None:
                wandb.run.summary["total_steps"] = total_steps
                wandb.run.summary["final_success_rate"] = final_metrics.get("eval_success_rate", 0)
                wandb.run.summary["improvement"] = final_metrics.get("eval_success_rate", 0) - initial_metrics.get("eval_success_rate", 0)

        print("Training complete!")
        
        return {
            "total_steps": total_steps,
            "training_rewards": training_rewards,
            "final_metrics": final_metrics if eval_dataset is not None else None,
            "initial_metrics": initial_metrics if eval_dataset is not None else None,
        }


def generate_with_cache(model, **kwargs):
    """
    Temporarily disables gradient checkpointing and enables caching for faster generation.
    """
    # Store original states
    was_gradient_checkpointing = model.is_gradient_checkpointing
    original_use_cache = model.config.use_cache

    # Disable gradient checkpointing and enable cache for generation
    if was_gradient_checkpointing:
        model.gradient_checkpointing_disable()
    model.config.use_cache = True

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(**kwargs)

    # Restore original states
    model.config.use_cache = original_use_cache
    if was_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return generated_ids


# ---------------------------------------------------------------------------
# Math environment using 24-game problems
# ---------------------------------------------------------------------------

def sample_math_batch(model, tokenizer, batch_size: int = 4, max_new_tokens: int = 512):
    """Generate a batch of math problems and model completions for GRPO training."""
    
    # Generate one math problem and use it for all batch elements
    problem_generator = generate_math_problems(tokenizer, 1)
    single_problem = next(problem_generator)
    problems = [single_problem for _ in range(batch_size)]
    
    # Extract prompts (all the same now)
    prompts = [problem["prompt"] for problem in problems]
    
    # Tokenize prompts
    tokenized = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    
    # Generate completions using the model
    generated = generate_with_cache(
        model,
        input_ids=input_ids.to(model.device),
        attention_mask=attention_mask.to(model.device),
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        repetition_penalty=1.1
    )
    
    # Extract only the generated parts (remove prompt)
    generated_tokens = generated[:, input_ids.shape[1]:]
    
    # Decode completions
    completions = [tokenizer.decode(gen, skip_special_tokens=True) for gen in generated_tokens]
    
    # Calculate rewards using math reward function
    rewards = torch.tensor(math_reward_func(completions, prompts), dtype=torch.float32)
    
    # Concatenate input_ids and generated_tokens for full sequences
    # Move input_ids to the same device as generated_tokens
    input_ids = input_ids.to(generated.device)
    full_sequences = generated
    
    # For GRPO, actions should only be the generated tokens (not prompt + completion)
    actions = generated_tokens.clone()
    
    # Store the prompt length (same for all items in batch since we use same problem)
    prompt_length = input_ids.shape[1]
    
    # Store the pad token ID for consistent usage
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    
    # Pad sequences to same length for batch processing
    max_full_len = max(seq.shape[0] for seq in full_sequences)
    max_action_len = max(seq.shape[0] for seq in generated_tokens)
    padded_input_ids = []
    padded_actions = []
    
    for i in range(batch_size):
        # Pad full sequences (for input_ids)
        full_seq_len = full_sequences[i].shape[0]
        if full_seq_len < max_full_len:
            padding = torch.full((max_full_len - full_seq_len,), pad_token_id, dtype=torch.long)
            padded_full_seq = torch.cat([full_sequences[i], padding])
        else:
            padded_full_seq = full_sequences[i][:max_full_len]
        
        # Pad generated tokens (for actions)
        action_seq_len = generated_tokens[i].shape[0]
        if action_seq_len < max_action_len:
            padding = torch.full((max_action_len - action_seq_len,), pad_token_id, dtype=torch.long)
            padded_action_seq = torch.cat([generated_tokens[i], padding])
        else:
            padded_action_seq = generated_tokens[i][:max_action_len]
        
        padded_input_ids.append(padded_full_seq)
        padded_actions.append(padded_action_seq)
    
    input_ids = torch.stack(padded_input_ids)
    actions = torch.stack(padded_actions)
    
    return input_ids, actions, rewards, prompt_length, pad_token_id


def evaluate_model(model, tokenizer, eval_dataset, max_new_tokens=512):
    """Simple evaluation function that generates completions and calculates rewards."""
    
    model.eval()
    total_reward = 0.0
    total_samples = 0
    success_count = 0
    
    # Process evaluation dataset
    for sample in eval_dataset:
        prompt = sample["query"]
        target = sample["target"]
        
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        # Generate completion
        generated = generate_with_cache(
            model,
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
        
        # Extract only the generated part
        prompt_length = inputs["input_ids"].shape[1]
        generated_tokens = generated[:, prompt_length:]
        completion = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        
        # Calculate reward
        rewards = math_reward_func([completion], [prompt])
        reward = rewards[0]
        
        total_reward += reward
        total_samples += 1
        if reward > 0:
            success_count += 1
    
    # Calculate metrics
    avg_reward = total_reward / total_samples if total_samples > 0 else 0.0
    success_rate = success_count / total_samples if total_samples > 0 else 0.0
    
    return {
        "eval_reward_mean": avg_reward,
        "eval_success_rate": success_rate,
        "eval_samples": total_samples
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="GRPO fine-tuning for Qwen3-1.7B on 24-game math problems")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="HuggingFace model identifier.",
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--steps", type=int, default=10, help="Number of optimisation steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (can be increased with LoRA)")
    parser.add_argument("--clip_ratio", type=float, default=0.2, help="PPO-style clip ratio")
    parser.add_argument("--kl_coef", type=float, default=0.01, help="KL penalty coefficient")
    parser.add_argument("--max_new_tokens", type=int, default=1200, help="Maximum new tokens to generate")
    parser.add_argument("--epochs_per_batch", type=int, default=4, help="Number of optimization steps per batch")
    
    # LoRA configuration
    parser.add_argument("--use_lora", action="store_true", default=True, help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    
    # Memory optimization
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing to save memory")
    
    # Wandb and evaluation configuration
    parser.add_argument("--use_wandb", action="store_true", default=True, help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="grpo-math-training", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default="custom-grpo", help="W&B run name")
    parser.add_argument("--eval_size", type=int, default=10, help="Number of problems for evaluation")
    
    # Reward normalization configuration
    parser.add_argument("--dr", action="store_true", default=True, help="Disable reward normalization (skip length normalization and std division)")
    
    # KL threshold configuration
    parser.add_argument("--kl_threshold", type=float, default=0.02, help="KL divergence threshold for early stopping")
    
    # Learning rate scheduler configuration
    parser.add_argument("--lr_schedule", action="store_true", default=True, help="Use linear learning rate decay")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Minimum learning rate as ratio of initial LR (default: 0.1 = 10% of initial LR)")

    args = parser.parse_args()

    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)  # Log all hyperparameters
        )

    # Load model & tokenizer (trust_remote_code required for Qwen series)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set padding side to 'left' for decoder-only models
    tokenizer.padding_side = 'left'
    
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # Enable gradient checkpointing for memory efficiency
    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing for memory optimization...")
        base_model.config.use_cache = False
        base_model.gradient_checkpointing_enable()

    # Apply LoRA by default (unless disabled)
    if args.use_lora:
        print("Applying LoRA for parameter-efficient fine-tuning...")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
        # With LoRA, the reference model is the frozen base model
        ref_model = base_model
    else:
        print("Training full model. This will consume more memory.")
        model = base_model
        # Without LoRA, we must create a deep copy for the reference model
        ref_model = copy.deepcopy(model)

    trainer = GRPOTrainer(
        model,
        ref_model=ref_model,
        lr=args.lr,
        clip_ratio=args.clip_ratio,
        kl_coef=args.kl_coef,
        dr=args.dr,
        total_steps=args.steps,
        lr_schedule=args.lr_schedule,
        min_lr_ratio=args.min_lr_ratio,
    )

    print("Starting GRPO fine-tuning with math problems …")
    if args.use_lora:
        print("Using LoRA for parameter-efficient fine-tuning.")
    else:
        print("Training full model. Consider using --use_lora for better memory efficiency.")
    
    # Create eval dataset once for reuse
    print(f"Creating evaluation dataset with {args.eval_size} problems...")
    
    # Generate data for evaluation
    eval_data = []
    problem_generator = generate_math_problems(tokenizer, args.eval_size)
    for problem in problem_generator:
        eval_data.append({
            "query": problem["prompt"],
            "target": problem["target"],
            "numbers": problem["numbers"]
        })
    
    eval_dataset = Dataset.from_list(eval_data)
    
    # Run training using the new train method
    training_results = trainer.train(
        tokenizer=tokenizer,
        steps=args.steps,
        epochs_per_batch=args.epochs_per_batch,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        eval_dataset=eval_dataset,
        use_wandb=args.use_wandb,
        kl_threshold=args.kl_threshold,
    )

    print("Training complete!")
    
    # Save model (LoRA adapters if using LoRA, full model otherwise)
    if args.use_lora:
        print("Saving LoRA adapters...")
        model.save_pretrained(f"./lora_adapters_grpo_math")
        print(f"LoRA adapters saved to ./lora_adapters_grpo_math")
    else:
        print("To save full model, use: model.save_pretrained('./saved_model')")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
