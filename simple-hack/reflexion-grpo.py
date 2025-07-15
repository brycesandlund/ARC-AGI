import argparse
import math
from typing import List, Dict, Any, Optional

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model

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

        Unlike PPO which clips the probability ratio, GRPO directly penalises
        the KL divergence while leaving the ratio un-clipped. This
        implementation keeps an optional PPO-style clip for stability that can
        be disabled by setting `clip_ratio <= 0.`"""
        ratio = torch.exp(new_logp - old_logp)
        if self.clip_ratio > 0:
            ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        # Policy gradient loss (reinforce)
        pg_loss = -(ratio * advantages).mean()
        # KL regularisation (old -> new)
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

        # Compute advantages with proper normalization (mean=0, std=1)
        advantages = rewards.unsqueeze(-1).expand_as(old_logp)
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
            input_ids=input_ids,
            attention_mask=attention_mask,
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
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    
    args = parser.parse_args()

    # Load model & tokenizer (trust_remote_code required for Qwen series)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA if requested
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
    total_steps = 0
    
    for episode in range(1, args.steps // args.epochs_per_batch + 1):
        # Sample once (expensive)
        batch = sample_math_batch(model, tokenizer, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
        input_ids, actions, rewards = batch
        
        # Compute old_logp ONCE from current policy
        with torch.no_grad():
            model.eval()
            outputs = model(input_ids.to(trainer.device))
            logits = outputs.logits[:, :-1, :]  # exclude final position
            target_actions = actions[:, 1:].to(trainer.device)  # actions correspond to next-tokens
            old_logp = trainer._old_log_probs(logits, target_actions)
        
        # Take multiple optimization steps using same old_logp
        for epoch in range(args.epochs_per_batch):
            total_steps += 1
            metrics = trainer.step(input_ids, actions, rewards, old_logp)
            
            print(
                f"Episode {episode:04d}, Epoch {epoch+1:02d}/{args.epochs_per_batch} | "
                f"loss: {metrics['loss']:.4f} | "
                f"approx_kl: {metrics['approx_kl']:.4f}"
            )
            
            # Early stopping if KL divergence gets too high
            if metrics['approx_kl'] > 0.02:
                print(f"  Early stopping due to high KL divergence: {metrics['approx_kl']:.4f}")
                break

    print("Training complete!")
    
    # Save model (LoRA adapters if using LoRA, full model otherwise)
    if args.use_lora:
        print("Saving LoRA adapters...")
        model.save_pretrained(f"./lora_adapters_grpo_math")
        print(f"LoRA adapters saved to ./lora_adapters_grpo_math")
    else:
        print("To save full model, use: model.save_pretrained('./saved_model')")


if __name__ == "__main__":
    main()
