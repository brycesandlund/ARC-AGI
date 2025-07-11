import argparse
import math
from typing import List, Dict, Any

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

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
    ) -> Dict[str, Any]:
        """Run a single optimisation step on a batch.

        Parameters
        ----------
        input_ids : (B, T) token ids fed into the model
        actions    : (B, T) actions sampled from the policy (identical to next tokens)
        rewards    : (B,) scalar reward for each sequence (broadcasted later)
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
        old_logp = self._old_log_probs(logits.detach(), target_actions)

        # Compute advantages (simple baseline ‑ mean).  You can plug in GAE or
        # any estimator of your choice here.
        advantages = rewards.unsqueeze(-1).expand_as(old_logp)
        advantages = advantages - advantages.mean()

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
# Dummy environment / reward (placeholder)
# ---------------------------------------------------------------------------

def sample_dummy_batch(tokenizer, batch_size: int = 4, max_len: int = 32):
    """Generate a batch of random prompts and returns tokenised tensors."""
    # For demonstration purposes we just pick random tokens (excluding special)
    vocab = list(range(tokenizer.vocab_size))
    special = set(tokenizer.all_special_ids)
    vocab = [tok for tok in vocab if tok not in special]

    prompts: List[List[int]] = []
    for _ in range(batch_size):
        length = int(torch.randint(4, max_len, (1,)).item())
        # Randomly sample `length` tokens from the non-special vocabulary
        indices = torch.randint(0, len(vocab), (int(length),))
        prompt = torch.tensor([vocab[i] for i in indices], dtype=torch.long)
        prompts.append(prompt.tolist())

    input_ids = [tokenizer.build_inputs_with_special_tokens(p) for p in prompts]
    input_ids = tokenizer.pad(
        {"input_ids": input_ids}, return_tensors="pt", padding=True
    )["input_ids"]

    # For this dummy example the "actions" are just the input shifted left.
    actions = input_ids.clone()

    # Reward: +1 if sequence length is > median else 0 (placeholder)
    rewards = (input_ids.sum(dim=-1) > input_ids.sum(dim=-1).median()).float()

    return input_ids, actions, rewards


# ---------------------------------------------------------------------------
# Main training loop (toy)
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="GRPO fine-tuning for Qwen3-1.7B")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="HuggingFace model identifier.",
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--steps", type=int, default=10, help="Number of optimisation steps")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--clip_ratio", type=float, default=0.2, help="PPO-style clip ratio")
    parser.add_argument("--kl_coef", type=float, default=0.01, help="KL penalty coefficient")
    args = parser.parse_args()

    # Load model & tokenizer (trust_remote_code required for Qwen series)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    trainer = GRPOTrainer(
        model,
        lr=args.lr,
        clip_ratio=args.clip_ratio,
        kl_coef=args.kl_coef,
    )

    print("Starting GRPO fine-tuning …")
    for step in range(1, args.steps + 1):
        batch = sample_dummy_batch(tokenizer, batch_size=args.batch_size)
        metrics = trainer.step(*batch)
        if step % 1 == 0:
            print(
                f"Step {step:04d}/{args.steps} | "
                f"loss: {metrics['loss']:.4f} | "
                f"approx_kl: {metrics['approx_kl']:.4f}"
            )

    print("Training complete. Save checkpoint with model.save_pretrained(…) if desired.")


if __name__ == "__main__":
    main()
