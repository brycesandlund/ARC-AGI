import argparse
import math
from typing import List, Dict, Any, Optional

import torch
from torch.nn import functional as F
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
        lr: float = 1e-5,
        clip_ratio: float = 0.2,
        kl_coef: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.clip_ratio = clip_ratio
        self.kl_coef = kl_coef
        self.device = device

    def _old_log_probs(self, logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return log-probabilities of `actions` under the policy that produced `logits`."""
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    def _policy_loss(
        self,
        new_logp: torch.Tensor,
        old_logp: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Core GRPO objective.
        
        GRPO uses a more direct policy gradient approach with proper KL regularization.
        Unlike PPO which clips ratios, GRPO relies on KL penalty for stability.
        """
        # Compute probability ratio (pi_new / pi_old)
        ratio = torch.exp(new_logp - old_logp)
        
        # Optional clipping for stability (can be disabled by setting clip_ratio <= 0)
        if self.clip_ratio > 0:
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            # Use the minimum of clipped and unclipped objectives
            pg_loss1 = -(ratio * advantages)
            pg_loss2 = -(clipped_ratio * advantages)
            pg_loss = torch.min(pg_loss1, pg_loss2).mean()
        else:
            # Pure GRPO without clipping
            pg_loss = -(ratio * advantages).mean()
        
        # Proper KL divergence: D_KL(π_old || π_new) = E[log(π_old) - log(π_new)]
        # This is reverse KL from old policy to new policy (standard in PPO/GRPO)
        kl_loss = (old_logp - new_logp).mean()
        
        return pg_loss + self.kl_coef * kl_loss

    def step(
        self,
        input_ids: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        old_logp: Optional[torch.Tensor] = None,  # Add this parameter
    ) -> Dict[str, Any]:
        """Run a single optimisation step on a batch.

        Parameters
        ----------
        input_ids : (B, T) token ids fed into the model
        actions    : (B, T) actions sampled from the policy (identical to next tokens)
        rewards    : (B,) scalar reward for each sequence (broadcasted later)
        old_logp   : (B, T-1) old log probabilities from original policy (optional)
        """
        self.model.train()
        input_ids = input_ids.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)

        # Forward pass
        outputs = self.model(input_ids)
        logits = outputs.logits[:, :-1, :]  # exclude final position (no next-token)
        target_actions = actions[:, 1:]      # actions correspond to next-tokens

        # Compute old log-probabilities (detach from graph)
        if old_logp is None:
            # First time - compute from current policy
            old_logp = self._old_log_probs(logits.detach(), target_actions)
        else:
            # Use provided old_logp from original policy
            old_logp = old_logp.to(self.device)

        # Compute advantages with sequence length normalization (mean=0, std=1)
        # Normalize rewards by sequence length first
        seq_lengths = (target_actions != 0).sum(dim=1).float().clamp(min=1.0)  # Assume 0 is pad token
        normalized_rewards = rewards / seq_lengths
        
        advantages = normalized_rewards.unsqueeze(-1).expand_as(old_logp)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # New log-probabilities for gradient flow
        new_logp = self._old_log_probs(logits, target_actions)

        # Compute GRPO loss
        loss = self._policy_loss(new_logp, old_logp, advantages)

        # Optimise
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        with torch.no_grad():
            approx_kl = (old_logp - new_logp).mean().item()

        return {
            "loss": loss.item(),
            "approx_kl": approx_kl,
        }


# ---------------------------------------------------------------------------
# Math environment using 24-game problems
# ---------------------------------------------------------------------------

def sample_math_batch(model, tokenizer, batch_size: int = 4, max_new_tokens: int = 512):
    """Generate a batch of math problems and model completions for GRPO training."""
    
    # Generate math problems
    problem_generator = generate_math_problems(tokenizer, batch_size)
    problems = [next(problem_generator) for _ in range(batch_size)]
    
    # Extract prompts
    prompts = [problem["prompt"] for problem in problems]
    
    # Tokenize prompts
    tokenized = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    
    # Generate completions using the model
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # Extract only the generated parts (remove prompt)
    prompt_lengths = input_ids.shape[1]
    generated_tokens = generated[:, prompt_lengths:]
    
    # Decode completions
    completions = [tokenizer.decode(gen, skip_special_tokens=True) for gen in generated_tokens]
    
    # Calculate rewards using math reward function
    rewards = torch.tensor(math_reward_func(completions, prompts), dtype=torch.float32)
    
    # Concatenate input_ids and generated_tokens for full sequences
    # Move input_ids to the same device as generated_tokens
    input_ids = input_ids.to(generated.device)
    full_sequences = torch.cat([input_ids, generated_tokens], dim=1)
    
    # For GRPO, actions are the full sequences (including prompt + completion)
    actions = full_sequences.clone()
    
    # Pad sequences to same length for batch processing
    max_len = max(seq.shape[0] for seq in full_sequences)
    padded_input_ids = []
    padded_actions = []
    
    for i in range(batch_size):
        seq_len = full_sequences[i].shape[0]
        if seq_len < max_len:
            # Pad with pad_token_id or eos_token_id
            pad_token = tokenizer.pad_token_id or tokenizer.eos_token_id
            padding = torch.full((max_len - seq_len,), pad_token, dtype=torch.long)
            padded_seq = torch.cat([full_sequences[i], padding])
        else:
            padded_seq = full_sequences[i][:max_len]
        
        padded_input_ids.append(padded_seq)
        padded_actions.append(padded_seq)
    
    input_ids = torch.stack(padded_input_ids)
    actions = torch.stack(padded_actions)
    
    return input_ids, actions, rewards


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
        with torch.no_grad():
            generated = model.generate(
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
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (can be increased with LoRA)")
    parser.add_argument("--clip_ratio", type=float, default=0.2, help="PPO-style clip ratio")
    parser.add_argument("--kl_coef", type=float, default=0.01, help="KL penalty coefficient")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--epochs_per_batch", type=int, default=4, help="Number of optimization steps per batch")
    
    # LoRA configuration
    parser.add_argument("--use_lora", action="store_true", default=True, help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    
    # Wandb and evaluation configuration
    parser.add_argument("--use_wandb", action="store_true", default=True, help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="grpo-math-training", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default="custom-grpo", help="W&B run name")
    parser.add_argument("--eval_size", type=int, default=10, help="Number of problems for evaluation")
    
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
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

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
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    trainer = GRPOTrainer(
        model,
        lr=args.lr,
        clip_ratio=args.clip_ratio,
        kl_coef=args.kl_coef,
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
    
    # Initial evaluation
    print("Running initial evaluation...")
    initial_metrics = evaluate_model(model, tokenizer, eval_dataset, args.max_new_tokens)
    print(f"Initial metrics: {initial_metrics}")

    if args.use_wandb:
        wandb.log({
            "initial_eval/success_rate": initial_metrics.get("eval_success_rate", 0),
            "initial_eval/reward_mean": initial_metrics.get("eval_reward_mean", 0),
            "episode": 0
        })

    total_steps = 0
    training_rewards = []
    
    for episode in range(1, args.steps // args.epochs_per_batch + 1):
        # Sample once (expensive)
        batch = sample_math_batch(model, tokenizer, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
        input_ids, actions, rewards = batch
        
        # Track batch rewards
        batch_reward_mean = rewards.mean().item()
        batch_reward_max = rewards.max().item()
        batch_success_rate = (rewards > 0).float().mean().item()
        training_rewards.extend(rewards.tolist())
        
        # Compute old_logp ONCE from current policy
        with torch.no_grad():
            # Keep model in train mode to match new_logp computation
            outputs = model(input_ids.to(trainer.device))
            logits = outputs.logits[:, :-1, :]  # exclude final position
            target_actions = actions[:, 1:].to(trainer.device)  # actions correspond to next-tokens
            old_logp = trainer._old_log_probs(logits, target_actions)
        
        # Take multiple optimization steps using same old_logp
        episode_losses = []
        episode_kls = []
        
        for epoch in range(args.epochs_per_batch):
            total_steps += 1
            metrics = trainer.step(input_ids, actions, rewards, old_logp)
            episode_losses.append(metrics['loss'])
            episode_kls.append(metrics['approx_kl'])
            
            # Log training metrics
            if args.use_wandb:
                wandb.log({
                    "train/loss": metrics['loss'],
                    "train/kl_divergence": metrics['approx_kl'],
                    "train/batch_reward_mean": batch_reward_mean,
                    "train/batch_reward_max": batch_reward_max,
                    "train/batch_success_rate": batch_success_rate,
                    "episode": episode,
                    "step": total_steps
                })
            
            print(
                f"Episode {episode:04d}, Epoch {epoch+1:02d}/{args.epochs_per_batch} | "
                f"loss: {metrics['loss']:.4f} | "
                f"kl: {metrics['approx_kl']:.4f} | "
                f"reward: {batch_reward_mean:.3f} | "
                f"success: {batch_success_rate:.1%}"
            )
            
            # Early stopping if KL divergence gets too high
            if metrics['approx_kl'] > 0.02:
                print(f"  Early stopping due to high KL divergence: {metrics['approx_kl']:.4f}")
                break

    # Final evaluation
    print("\nRunning final evaluation...")
    final_metrics = evaluate_model(model, tokenizer, eval_dataset, args.max_new_tokens)
    print(f"Final metrics: {final_metrics}")

    if args.use_wandb:
        wandb.log({
            "final_eval/success_rate": final_metrics.get("eval_success_rate", 0),
            "final_eval/reward_mean": final_metrics.get("eval_reward_mean", 0),
            "episode": episode
        })
        
        # Log training summary
        if wandb.run is not None:
            wandb.run.summary["total_steps"] = total_steps
            wandb.run.summary["final_success_rate"] = final_metrics.get("eval_success_rate", 0)
            wandb.run.summary["improvement"] = final_metrics.get("eval_success_rate", 0) - initial_metrics.get("eval_success_rate", 0)

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
