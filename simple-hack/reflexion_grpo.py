import argparse
import math
import random
import time
from typing import List, Dict, Any, Optional, Tuple
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
from reflexion_grpo_tests import test_sample, debug_batch_and_actions

# Global token IDs - initialized in main() after tokenizer is loaded
PAD_TOKEN_ID = None
EOS_TOKEN_ID = None
SEQUENCE_LENGTH_NORMALIZATION = 1000.0

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
        total_steps: Optional[int] = None,
        lr_schedule: bool = True,
        min_lr_ratio: float = 0.1,
        grad_clip_norm: Optional[float] = 1.0,
    ) -> None:
        self.model = model.to(device)
        self.ref_model = ref_model.to(device)
        self.ref_model.eval()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.clip_ratio = clip_ratio
        self.kl_coef = kl_coef
        self.device = device
        self.grad_clip_norm = grad_clip_norm
        
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
        
        # Print gradient clipping configuration
        if self.grad_clip_norm is not None:
            print(f"Gradient clipping enabled with norm: {self.grad_clip_norm}")
        else:
            self.grad_clip_norm = float('inf')
            print("Gradient clipping disabled")

    def _compute_log_probs(self, model: torch.nn.Module, input_ids: torch.Tensor, disable_adapter: bool = False) -> torch.Tensor:
        """Computes log probabilities for a given model and input_ids.

        Args:
            model: The model to be used for computation.
            input_ids: The input tensor for the model of shape (batch_size, sequence_length).
            disable_adapter: A boolean flag to disable the adapter if it exists.
        
        Returns:
            A tensor of log probabilities of shape (batch_size, sequence_length - 1).
        """
        input_ids = input_ids.to(self.device)
        attention_mask = (input_ids != PAD_TOKEN_ID).long()
        target_actions = input_ids[:, 1:]

        # We must actually disable the adapter on self.model, ignoring the reference model.
        if disable_adapter and hasattr(self.model, "disable_adapter"):
            with self.model.disable_adapter():
                self._disable_dropout(self.model)
                outputs = self.model(input_ids, attention_mask=attention_mask)
        else:
            self._disable_dropout(self.model)
            outputs = model(input_ids, attention_mask=attention_mask)
        
        logits = outputs.logits[:, :-1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(-1, target_actions.unsqueeze(-1)).squeeze(-1)

    def _pg_loss(
        self,
        new_logp: torch.Tensor,
        old_logp: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """Computes the policy gradient loss component."""
        ratio = torch.exp(new_logp - old_logp)
        clipped_fraction = 0.0
        if self.clip_ratio > 0:
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            pg_loss1 = -(ratio * advantages)
            pg_loss2 = -(clipped_ratio * advantages)
            pg_losses = torch.max(pg_loss1, pg_loss2)
            
            # Count clipping statistics
            clipped_mask = (ratio < 1.0 - self.clip_ratio) | (ratio > 1.0 + self.clip_ratio)
            clipped_fraction = clipped_mask.float().mean().item()
            
            return pg_losses.sum(), clipped_fraction
        else:
            unclipped_losses = -(ratio * advantages)
            return unclipped_losses.sum(), clipped_fraction

    def _kl_loss(
        self,
        new_logp: torch.Tensor,
        ref_logp: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the KL divergence loss against the reference model."""
        log_ratio_ref = ref_logp - new_logp
        ratio_ref = torch.exp(log_ratio_ref)
        kl_losses = ratio_ref - log_ratio_ref - 1
        return kl_losses

    def _disable_dropout(self, model):
        """Sets all dropout layers to eval mode for the given model."""
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.eval()

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        old_logp_full: torch.Tensor,
        advantages_per_sequence: torch.Tensor,
        rollouts_per_prompt: int,
        is_first_step: bool = False,
    ) -> Dict[str, Any]:
        """Compute the loss for a batch, but do not perform an optimization step.
        This is used for gradient accumulation.

        Parameters
        ----------
        input_ids : (B, T) full sequence tokens (prompt + generated)
        loss_mask  : (B, T-1) mask indicating which tokens are part of the sequence to compute loss for
        old_logp_full   : (B, T-1) full old log probabilities from the policy before optimization.
        advantages_per_sequence : (B,) single scalar advantage per sequence
        rollouts_per_prompt : number of rollouts to normalize loss by
        """

        # # Debug: Print shapes of input tensors
        # print(f"Debug - Input tensor shapes:")
        # print(f"  input_ids: {input_ids.shape}")
        # print(f"  loss_mask: {loss_mask.shape}")
        # print(f"  old_logp_full: {old_logp_full.shape}")
        # print(f"  advantages_per_sequence: {advantages_per_sequence.shape}")

        # Debug: Print advantages
        # print(f"advantages_per_sequence: {advantages_per_sequence}")

        input_ids = input_ids.to(self.device)
        loss_mask = loss_mask.to(self.device)

        # Compute reference log-probs
        with torch.no_grad():
            ref_logp_full = self._compute_log_probs(
                model=self.ref_model,
                input_ids=input_ids,
                disable_adapter=True,
            )
            ref_logp = ref_logp_full[loss_mask]

        # Compute old log-probs from full distribution
        old_logp_full = old_logp_full.to(self.device)
        old_logp = old_logp_full[loss_mask] # Apply mask

        # Advantages are provided per-sequence; expand to per-token via repeat_interleave
        seq_advantages = advantages_per_sequence.to(self.device)

        # For each sequence, repeat its advantage for the number of generated tokens included in the loss
        token_counts_per_sequence = loss_mask.sum(dim=1)
        advantages = torch.repeat_interleave(seq_advantages, token_counts_per_sequence)
        
        # New log-probabilities for gradient flow
        new_logp_full = self._compute_log_probs(self.model, input_ids)
        new_logp = new_logp_full[loss_mask] # Apply mask

        # Calculate model entropy over the generated tokens for logging
        with torch.no_grad():
            estimated_entropy = -new_logp.detach()
            mean_entropy = estimated_entropy.mean().item()

        # Compute loss components
        pg_loss, clipped_fraction = self._pg_loss(new_logp, old_logp, advantages)

        # # Debug: Check if new_logp equals ref_logp
        # logp_equal = torch.allclose(new_logp, ref_logp, atol=1e-6)
        # print(f"new_logp equals ref_logp: {logp_equal}")

        # # Debug: Check if new_logp equals old_logp
        # logp_equal = torch.allclose(new_logp, old_logp, atol=1e-6)
        # print(f"new_logp equals old_logp: {logp_equal}")
        
        # # If this is the first step and logp tensors don't agree, print them
        # if is_first_step and not logp_equal:
        #     print(f"First step - old_logp and new_logp do not agree:")
        #     print(f"old_logp: {old_logp}")
        #     print(f"new_logp: {new_logp}")

        kl_loss = self._kl_loss(new_logp, ref_logp)
        loss = (pg_loss + self.kl_coef * kl_loss.sum()) / SEQUENCE_LENGTH_NORMALIZATION / float(rollouts_per_prompt)

        avg_response_length = loss_mask.sum().item() / loss_mask.shape[0]

        return {
            "loss": loss,
            "pg_loss": pg_loss.item(),
            "kl_loss": kl_loss.mean().item(),
            "clipped_fraction": clipped_fraction,
            "avg_response_length": avg_response_length,
            "model_entropy": mean_entropy,
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
                f"{eval_type}_eval/reward_std": metrics.get("eval_reward_std", 0),
                f"{eval_type}_eval/avg_response_length": metrics.get("eval_avg_response_length", 0),
                "episode": episode
            })
        
        return metrics

    def train(
        self,
        tokenizer: Any,
        collection_steps: int,
        batch_size: int,
        epochs_per_batch: int,
        rollouts_per_prompt: int,
        prompts_per_generation: int,
        max_new_tokens: int,
        eval_dataset: Optional[Any] = None,
        use_wandb: bool = False,
        kl_threshold: float = 0.02,
        use_revision: bool = False,
        minibatch_size: int = 1,
    ) -> Dict[str, Any]:
        """Train the model using GRPO.
        
        Parameters
        ----------
        tokenizer : tokenizer for the model
        steps : total number of optimization steps
        batch_size : Number of prompts to use for experience generation in each collection step.
        epochs_per_batch : number of optimization epochs to run on each collected batch of experience
        rollouts_per_prompt : number of rollouts to generate for each prompt
        max_new_tokens : maximum new tokens to generate
        eval_dataset : optional dataset for evaluation
        use_wandb : whether to log to wandb
        kl_threshold : KL divergence threshold for early stopping
        use_revision : whether to use revision model for sampling
        minibatch_size : size of minibatches for optimization
        
        Returns
        -------
        Dict with training statistics
        """
        total_optim_steps = 0
        training_rewards = []
        
        # Initial evaluation if eval dataset provided
        if eval_dataset is not None:
            initial_metrics = self._run_evaluation(eval_dataset, tokenizer, max_new_tokens, "initial", 0, use_wandb)

        print("Starting GRPO training...")
        
        # Main training loop (steps are now data collection cycles)
        for collection_step in range(1, collection_steps + 1):
            
            # --- 1. Data Collection Phase ---
            collection_start_time = time.time()
            experience_buffer = []
            step_rewards_mean = []
            step_rewards_max = []
            step_success_rates = []

            print(f"\nCollecting experience for collection step {collection_step}/{collection_steps}...")
            self.model.eval()
            for micro_step in range(batch_size // prompts_per_generation):
                # Sample a fresh batch for each accumulation step
                if use_revision:
                    sample_start_time = time.time()
                    batch = sample_and_revise(
                        model=self.model,
                        tokenizer=tokenizer,
                        revision_model=self.model,
                        rollouts_per_prompt=rollouts_per_prompt,
                        prompts_per_generation=prompts_per_generation,
                        max_new_tokens=max_new_tokens,
                        disable_adapter=False,
                        enable_thinking=False
                    )
                    sample_time = time.time() - sample_start_time
                    print(f"  sample_and_revise took {sample_time:.2f}s")
                else:
                    sample_start_time = time.time()
                    batch = sample(self.model, tokenizer, rollouts_per_prompt=rollouts_per_prompt, prompts_per_generation=prompts_per_generation, max_new_tokens=max_new_tokens)
                    sample_time = time.time() - sample_start_time
                    print(f"  sample took {sample_time:.2f}s")
                
                input_ids, rewards, advantages, loss_mask, _, _ = batch

                # Split tensors
                input_ids_chunks = torch.split(input_ids, rollouts_per_prompt)
                rewards_chunks = torch.split(rewards, rollouts_per_prompt)
                advantages_chunks = torch.split(advantages, rollouts_per_prompt)
                loss_mask_chunks = torch.split(loss_mask, rollouts_per_prompt)

                for i in range(prompts_per_generation):
                    with torch.no_grad():
                        chunk_logp = self._compute_log_probs(self.model, input_ids_chunks[i])

                    experience_buffer.append({
                        'input_ids': input_ids_chunks[i].to('cpu'), 
                        'rewards': rewards_chunks[i].to('cpu'),
                        'advantages': advantages_chunks[i].to('cpu'),
                        'loss_mask': loss_mask_chunks[i].to('cpu'),
                        'old_logp_full': chunk_logp.detach().to('cpu')
                    })
                
                # Track and log rewards from this collection micro-batch
                batch_reward_mean = rewards.mean().item()
                training_rewards.extend(rewards.tolist())
                step_rewards_mean.append(batch_reward_mean)
                step_rewards_max.append(rewards.max().item())
                step_success_rates.append((rewards > 0).float().mean().item())
                print(f"  Collected micro-batch {micro_step+1}/{batch_size // prompts_per_generation} | reward: {batch_reward_mean:.3f}")
            
            # Clear GPU cache after all experience collection is complete
            print("Clearing GPU cache after experience collection...")
            torch.cuda.empty_cache()

            collection_time = time.time() - collection_start_time
            print(f"Data collection for step {collection_step} took {collection_time:.2f}s")
            if use_wandb:
                wandb.log({"train/data_collection_time": collection_time, "collection_step": collection_step})

            # --- 2. Optimization Phase ---
            optimization_start_time = time.time()
            self.model.train()
            kl_exceeded = False
            is_first_gradient_step = True
            for epoch in range(epochs_per_batch):
                random.shuffle(experience_buffer)  # Shuffle experience for each epoch
                
                # Process in minibatches
                for i in range(0, len(experience_buffer), minibatch_size):
                    minibatch = experience_buffer[i:i+minibatch_size]
                    
                    self.optimizer.zero_grad()
                    
                    minibatch_losses = []
                    minibatch_pg_losses = []
                    minibatch_kls = []
                    minibatch_clipped_fractions = []
                    minibatch_response_lengths = []
                    minibatch_entropies = []
                    non_zero_advantages_count = 0

                    # Iterate over the collected experience in the minibatch
                    for micro_batch_data in minibatch:

                        # debug_batch_and_actions(
                        #     tokenizer,
                        #     micro_batch_data['input_ids'],
                        #     micro_batch_data['loss_mask'],
                        #     context="Training Minibatch"
                        # )

                        # Compute loss for the micro-batch using the pre-computed old_logp
                        compute_loss_start_time = time.time()
                        metrics = self.compute_loss(
                            input_ids=micro_batch_data['input_ids'],
                            loss_mask=micro_batch_data['loss_mask'],
                            old_logp_full=micro_batch_data['old_logp_full'],
                            advantages_per_sequence=micro_batch_data['advantages'],
                            rollouts_per_prompt=rollouts_per_prompt,
                            is_first_step=is_first_gradient_step,
                        )
                        compute_loss_time = time.time() - compute_loss_start_time
                        print(f"    compute_loss took {compute_loss_time:.3f}s, advantages non-zero: {torch.any(micro_batch_data['advantages'] != 0).item()}")
                        
                        # Track advantages for logging
                        if torch.any(micro_batch_data['advantages'] != 0):
                            non_zero_advantages_count += 1

                        loss = metrics['loss']
                        
                        # Accumulate gradients
                        loss.backward()

                        # Store metrics for logging
                        minibatch_losses.append(loss.item())
                        minibatch_pg_losses.append(metrics['pg_loss'])
                        minibatch_kls.append(metrics['kl_loss'])
                        minibatch_clipped_fractions.append(metrics['clipped_fraction'])
                        minibatch_response_lengths.append(metrics['avg_response_length'])
                        minibatch_entropies.append(metrics['model_entropy'])
                    
                    # Aggregate KL divergence from the minibatch
                    avg_kl = sum(minibatch_kls) / len(minibatch_kls) if minibatch_kls else 0.0

                    # Check KL threshold before optimizer step
                    if avg_kl > kl_threshold:
                        print(f"  KL divergence {avg_kl:.4f} exceeds threshold {kl_threshold}. Abandoning optimization for this batch.")
                        kl_exceeded = True
                        break

                    # Clip gradients and perform optimizer step after accumulating over the whole minibatch
                    total_optim_steps += 1
                    
                    # Gradient Norm Calculation and Clipping. clip_grad_norm_ returns the total norm of
                    # all parameters (viewed as a single vector) before clipping.
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm).item()

                    self.optimizer.step()
                    
                    # After the first optimizer step, set the flag to False
                    is_first_gradient_step = False
                    
                    # Capture LR before scheduler step for accurate logging
                    current_lr = self.optimizer.param_groups[0]['lr']
                    
                    # Step the learning rate scheduler
                    if self.scheduler is not None:
                        self.scheduler.step()
                        
                    # Log aggregated metrics for the optimization step
                    avg_loss = sum(minibatch_losses) / len(minibatch_losses)
                    avg_pg_loss = sum(minibatch_pg_losses) / len(minibatch_pg_losses)
                    avg_clipped_fraction = sum(minibatch_clipped_fractions) / len(minibatch_clipped_fractions)
                    avg_response_length = sum(minibatch_response_lengths) / len(minibatch_response_lengths)
                    avg_entropy = sum(minibatch_entropies) / len(minibatch_entropies)
                    
                    # Compute reward metrics from the current minibatch
                    minibatch_rewards = torch.cat([data['rewards'] for data in minibatch])
                    avg_reward_mean = minibatch_rewards.mean().item()
                    avg_reward_max = minibatch_rewards.max().item()
                    avg_reward_std = minibatch_rewards.std().item()
                    avg_success_rate = (minibatch_rewards > 0).float().mean().item()
                    
                    fraction_non_zero_advantages = non_zero_advantages_count / len(minibatch) if minibatch else 0.0

                    if use_wandb:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/pg_loss": avg_pg_loss,
                            "train/kl_divergence": avg_kl,
                            "train/clipped_fraction": avg_clipped_fraction,
                            "train/avg_response_length": avg_response_length,
                            "train/learning_rate": current_lr,
                            "train/batch_reward_mean": avg_reward_mean,
                            "train/batch_reward_max": avg_reward_max,
                            "train/batch_reward_std": avg_reward_std,
                            "train/batch_success_rate": avg_success_rate,
                            "train/model_entropy": avg_entropy,
                            "train/grad_norm": grad_norm,
                            "train/fraction_non_zero_advantages": fraction_non_zero_advantages,
                            "step": total_optim_steps,
                            "collection_step": collection_step,
                            "epoch_per_batch": epoch + 1,
                        })
                    
                    minibatch_num = i // minibatch_size + 1
                    total_minibatches = math.ceil(len(experience_buffer) / minibatch_size)
                    
                    print(
                        f"Optim Step {total_optim_steps:05d} | Collection Step {collection_step}/{collection_steps}, Epoch {epoch+1}/{epochs_per_batch}, MiniBatch {minibatch_num}/{total_minibatches} | "
                        f"loss: {avg_loss:.4f} | "
                        f"kl: {avg_kl:.4f} | "
                        f"clipped: {avg_clipped_fraction:.3f} | "
                        f"len: {avg_response_length:.1f} | "
                        f"lr: {current_lr:.2e} | "
                        f"reward: {avg_reward_mean:.3f} | "
                        f"reward_std: {avg_reward_std:.3f} | "
                        f"grad_norm: {grad_norm:.4f} | "
                        f"success: {avg_success_rate:.1%} | "
                        f"entropy: {avg_entropy:.4f}"
                    )
                
                if kl_exceeded:
                    break

            optimization_time = time.time() - optimization_start_time
            print(f"Optimization for step {collection_step} took {optimization_time:.2f}s")
            if use_wandb:
                wandb.log({"train/optimization_time": optimization_time, "collection_step": collection_step})

        # Final evaluation if eval dataset provided
        if eval_dataset is not None:
            final_metrics = self._run_evaluation(eval_dataset, tokenizer, max_new_tokens, "final", total_optim_steps, use_wandb)
                
            # Log training summary
            if use_wandb and wandb.run is not None:
                wandb.run.summary["total_steps"] = total_optim_steps
                wandb.run.summary["final_success_rate"] = final_metrics.get("eval_success_rate", 0)
                wandb.run.summary["improvement"] = final_metrics.get("eval_success_rate", 0) - initial_metrics.get("eval_success_rate", 0)

        print("Training complete!")
        
        return {
            "total_steps": total_optim_steps,
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
    with torch.inference_mode():
        generated_ids = model.generate(**kwargs)

    # Restore original states
    model.config.use_cache = original_use_cache
    if was_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return generated_ids


def _extract_completions_and_create_loss_mask(tokenizer, generated_ids: torch.Tensor, input_ids: torch.Tensor) -> Tuple[List[str], torch.Tensor]:
    """Decodes generated token sequences, extracts completions, and creates a loss mask.

    Mask construction:
    - Computes a single-row mask of shape (1, T-1) where positions >= prompt_length-1 are True.
    - Expands this mask to (B, T-1) to match the batch size for downstream boolean indexing.
    - Excludes PAD tokens from the loss calculation.
    """
    # Slice the generated_ids tensor to get only the tokens that were generated after the prompt.
    completion_ids = generated_ids[:, input_ids.shape[1]:]
    
    # Decode the completion tokens, skipping special tokens.
    completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

    # Create the loss mask.
    prompt_len = input_ids.shape[1]
    seq_len = generated_ids.shape[1]
    device = generated_ids.device
    
    # The loss is calculated on logits, which have a sequence length of T-1.
    # The mask should be True for all positions >= prompt_length - 1.
    positions = torch.arange(seq_len - 1, device=device).unsqueeze(0)  # Shape: (1, seq_len - 1)
    position_mask = positions >= (prompt_len - 1)  # (1, T-1)
    position_mask = position_mask.expand(generated_ids.size(0), -1)  # (B, T-1)

    # Also mask out pad tokens.
    not_pad_mask = (generated_ids[:, 1:] != PAD_TOKEN_ID)
    loss_mask = position_mask & not_pad_mask
    
    return completions, loss_mask


def compute_sequence_advantages(rewards: torch.Tensor, prompts_per_generation: int, rollouts_per_prompt: int, eps: float = 1e-8) -> torch.Tensor:
    """Compute simple per-sequence advantages by normalizing rewards.
    Advantages are computed on a per-prompt basis.

    advantages = (rewards - mean) / (std + eps)

    Parameters
    ----------
    rewards : torch.Tensor
        Tensor of shape (B,) with scalar rewards per sequence
    prompts_per_generation : int
        Number of unique prompts per generation
    rollouts_per_prompt : int
        Number of rollouts per prompt
    eps : float
        Small value to avoid division by zero

    Returns
    -------
    torch.Tensor
        Tensor of shape (B,) with a single advantage per sequence
    """
    rewards = rewards.float()
    
    # Reshape rewards to (prompts_per_generation, rollouts_per_prompt)
    rewards_per_prompt = rewards.view(prompts_per_generation, rollouts_per_prompt)
    
    # Compute mean and std per prompt
    mean_per_prompt = rewards_per_prompt.mean(dim=1, keepdim=True)
    std_per_prompt = rewards_per_prompt.std(dim=1, keepdim=True)
    
    # Normalize rewards per prompt
    advantages_per_prompt = (rewards_per_prompt - mean_per_prompt) / (std_per_prompt + eps)
    
    # Flatten advantages back to (B,)
    return advantages_per_prompt.view(-1)


def generate_and_decode(model, tokenizer, prompts, max_new_tokens, disable_adapter=False, enable_thinking: bool = True, **gen_kwargs):
    """Generates completions from a model and decodes them.

    Parameters
    ----------
    model : torch.nn.Module
        The autoregressive language model used for generation.
    tokenizer : Any
        The tokenizer used to build chat prompts and decode outputs.
    prompts : List[str]
        User prompts to generate completions for.
    max_new_tokens : int
        Maximum number of new tokens to generate per sequence.
    disable_adapter : bool, optional
        If True, temporarily disables PEFT adapters during generation when supported.
    enable_thinking : bool, optional
        If True, enables the chat template's "thinking" mode when available.
    **gen_kwargs : Any
        Additional keyword arguments forwarded to `model.generate()`.

    Returns
    -------
    Tuple[List[str], torch.Tensor, torch.Tensor]
        - completions (List[str]): Decoded completions for each prompt with special
          tokens removed. Length equals `len(prompts)`.
        - generated_ids (torch.Tensor): Token IDs for the full sequences
          (prompt + completion), left-padded to a common length. Shape:
          `(batch_size, seq_len)`.
        - loss_mask (torch.Tensor): A boolean mask that is `True` only for the
          generated (completion) tokens.
          Shape: `(batch_size, seq_len - 1)`.
    """
    
    # Convert prompts to chat message format
    messages_list = [[{"role": "user", "content": p}] for p in prompts]
    
    # # Debug: Print prompts for debugging
    # print("Debug: Prompts being processed:")
    # for i, prompt in enumerate(prompts):
    #     print(f"Prompt {i}: {prompt}")
    # print()
    
    # Apply chat template with thinking mode control
    # This creates the full prompt string including special tokens
    processed_prompts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True, # Note: Transformers recommends not using this during training but you get gibberish w/o it
            enable_thinking=enable_thinking
        ) for messages in messages_list
    ]

    tokenized = tokenizer(processed_prompts, return_tensors="pt", padding=True, truncation=True)
    
    # Base generation arguments
    base_gen_kwargs = {
        "input_ids": tokenized["input_ids"].to(model.device),
        "attention_mask": tokenized["attention_mask"].to(model.device),
        "max_new_tokens": max_new_tokens,
        "temperature": 0.6,
        "do_sample": True,
        "pad_token_id": PAD_TOKEN_ID,
        "repetition_penalty": 1.1,
    }
    # Update with any additional kwargs
    base_gen_kwargs.update(gen_kwargs)
    
    # Handle adapter disabling for PEFT models
    if disable_adapter and hasattr(model, "disable_adapter"):
        with model.disable_adapter():  # type: ignore[attr-defined]
            generated_ids = generate_with_cache(model, **base_gen_kwargs)
    else:
        generated_ids = generate_with_cache(model, **base_gen_kwargs)

    # Debug: Check for padding and EOS tokens in generated sequences
    is_pad_tensor = (generated_ids == PAD_TOKEN_ID)
    is_eos_tensor = (generated_ids == EOS_TOKEN_ID)
    
    # torch.set_printoptions(threshold=10_000, linewidth=200)
    # print("Generated IDs shape:", generated_ids.shape)
    # print("Pad token ID:", PAD_TOKEN_ID)
    # print("EOS token ID:", EOS_TOKEN_ID)
    # print("is_pad_tensor (generated_ids == PAD_TOKEN_ID):")
    # print(is_pad_tensor)
    # print("is_eos_tensor (generated_ids == EOS_TOKEN_ID):")
    # print(is_eos_tensor)
    # print(tokenizer.batch_decode(generated_ids))
        
    # Extract, decode, and return completions using token-based slicing.
    completions, loss_mask = _extract_completions_and_create_loss_mask(tokenizer, generated_ids, tokenized["input_ids"])
    
    # Return text completions, full generated sequence, and the loss mask
    return completions, generated_ids, loss_mask


def sample_and_revise(
    model, 
    tokenizer, 
    revision_model,
    rollouts_per_prompt: int, 
    prompts_per_generation: int,
    max_new_tokens: int, 
    disable_adapter: bool, 
    enable_thinking: bool
):
    """Samples, revises, and evaluates completions.

    This function performs a two-stage generation process:
    1. It first calls `sample()` to generate initial completions from a base model.
    2. It then constructs new prompts that ask a `revision_model` to revise the
       initial solutions.
    3. It generates revised completions, calculates their rewards and advantages,
       and returns a batch ready for training.

    This is a form of self-improvement where the model refines its own output.

    Parameters
    ----------
    model : torch.nn.Module
        The base model for generating initial solutions.
    tokenizer : Any
        The tokenizer for encoding and decoding.
    revision_model : torch.nn.Module
        The model used to revise the initial solutions.
    rollouts_per_prompt : int
        Number of completions to generate per unique problem.
    prompts_per_generation : int
        Number of unique problems to generate.
    max_new_tokens : int
        Maximum number of new tokens for both initial and revised generation.
    disable_adapter : bool
        If True, disables the PEFT adapter during generation (if applicable).
    enable_thinking : bool
        If True, enables "thinking" mode in the prompt template, which can affect
        how the model formats its chain-of-thought output.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[str]]
        A tuple containing the data from the *revised* completions:
        - input_ids (torch.Tensor): The full token sequences of the *revised*
          completions (revision prompt + revised completion), padded.
          Shape: `(prompts_per_generation * rollouts_per_prompt, padded_revised_sequence_length)`
        - rewards (torch.Tensor): A scalar reward for each *revised* sequence.
          Shape: `(prompts_per_generation * rollouts_per_prompt,)`
        - advantages (torch.Tensor): A normalized advantage for each *revised* sequence.
          Shape: `(prompts_per_generation * rollouts_per_prompt,)`
        - loss_mask (torch.Tensor): A boolean mask that is `True` only for the
          tokens in the *revised* completion.
          Shape: `(prompts_per_generation * rollouts_per_prompt, padded_revised_sequence_length - 1)`
        - prompts (List[str]): The list of original prompts (from the first pass).
        - final_completions (List[str]): The list of decoded *revised* completions.
    """
    # 1. First pass: Sample from the base model to get initial solutions
    _, _, _, _, prompts, initial_completions = sample(
        model, tokenizer, rollouts_per_prompt, prompts_per_generation, max_new_tokens
    )

    # 2. Second pass: Construct revision prompts and revise with the revision_model
    revision_prompts = []
    for i in range(len(prompts)):
        full_sequence_text = prompts[i] + initial_completions[i]
        # For revision, we can just use the initial completion's reward, though it's not strictly necessary.
        # Here, we will just pass a placeholder since the prompt is about revision.
        # A more advanced implementation could use the reward to guide revision.
                
        revision_prompt = f"""The following is a solution to a math problem.
Problem and solution:
"{full_sequence_text}"

Your task is to revise the chain-of-thought (content in <think> tags) to be more concise and possibly change/complete the final answer. Keep all tokens in the chain-of-thought that are helpful to achieving the correct answer. Eliminate dead ends.

The revised completion should be in the format: <think>chain-of-thought</think> answer.
"""
        revision_prompts.append(revision_prompt)
    
    # Generate revised completions
    revised_completions, revised_generated_ids, loss_mask = generate_and_decode(
        revision_model,
        tokenizer,
        revision_prompts,
        max_new_tokens,
        disable_adapter=disable_adapter,
        enable_thinking=enable_thinking,
    )

    # Use revised final answers only if thinking mode is enabled, otherwise use original completions
    if enable_thinking:
        # Extract just the final answer (not the thinking content) from revised completions
        final_completions = [parse_completion(completion)[1] for completion in revised_completions]
    else:
        final_completions = revised_completions
    
    # Create the batch from prompts and generated completions
    rewards = torch.tensor(math_reward_func(final_completions, prompts), dtype=torch.float32)
    advantages = compute_sequence_advantages(rewards, prompts_per_generation, rollouts_per_prompt)
    input_ids = revised_generated_ids

    # The loss mask is now directly returned from generate_and_decode
    
    return input_ids, rewards, advantages, loss_mask, prompts, final_completions


def sample(model, tokenizer, rollouts_per_prompt: int = 4, prompts_per_generation: int = 1, max_new_tokens: int = 512):
    """Generate a batch of math problems and model completions for GRPO training.

    This function first generates a set of unique math problems, then duplicates them
    to create a batch of prompts for generating multiple rollouts. It then uses
    the model to generate completions for these prompts. Finally, it computes
    rewards and advantages for the generated sequences.

    Parameters
    ----------
    model : torch.nn.Module
        The model to use for generating completions.
    tokenizer : Any
        The tokenizer for encoding prompts and decoding completions.
    rollouts_per_prompt : int, optional
        Number of completions to generate for each unique problem, by default 4.
    prompts_per_generation : int, optional
        Number of unique math problems to generate, by default 1.
    max_new_tokens : int, optional
        Maximum number of new tokens to generate for each completion, by default 512.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[str]]
        A tuple containing:
        - input_ids (torch.Tensor): The full token sequences (prompt + completion),
          padded to the same length.
          Shape: `(prompts_per_generation * rollouts_per_prompt, padded_sequence_length)`
        - rewards (torch.Tensor): A scalar reward for each generated sequence.
          Shape: `(prompts_per_generation * rollouts_per_prompt,)`
        - advantages (torch.Tensor): A normalized advantage value for each sequence.
          Shape: `(prompts_per_generation * rollouts_per_prompt,)`
        - loss_mask (torch.Tensor): A boolean mask that is `True` only for the
          generated (completion) tokens.
          Shape: `(prompts_per_generation * rollouts_per_prompt, padded_sequence_length - 1)`
        - prompts (List[str]): The list of prompts used for generation.
        - completions (List[str]): The list of decoded model completions.
    """
    
    # Generate `prompts_per_generation` unique math problems
    problem_generator = generate_math_problems(tokenizer, prompts_per_generation)
    unique_problems = list(problem_generator)
    
    # Duplicate each problem `rollouts_per_prompt` times
    problems = [problem for problem in unique_problems for _ in range(rollouts_per_prompt)]
    
    # Extract prompts
    prompts = [problem["prompt"] for problem in problems]

    # Generate completions using the model
    completions, generated_ids, loss_mask = generate_and_decode(model, tokenizer, prompts, max_new_tokens, enable_thinking=True)
    
    # Create the batch from prompts and generated completions
    rewards = torch.tensor(math_reward_func(completions, prompts), dtype=torch.float32)
    advantages = compute_sequence_advantages(rewards, prompts_per_generation, rollouts_per_prompt)
    input_ids = generated_ids

    # test_sample(input_ids, rewards, advantages, loss_mask, prompts, completions, rollouts_per_prompt, prompts_per_generation)
    # debug_batch_and_actions(tokenizer, input_ids, loss_mask, context="Data Sampling")

    return input_ids, rewards, advantages, loss_mask, prompts, completions


def evaluate_model(model, tokenizer, eval_dataset, max_new_tokens=512, batch_size=12):
    """Batched evaluation that uses the same chat templating and generation path as training."""

    model.eval()
    total_reward = 0.0
    total_samples = 0
    success_count = 0
    total_response_length = 0.0
    all_rewards = []

    # Convert dataset to list if it's not already
    eval_samples = list(eval_dataset)

    # Process evaluation dataset in batches
    for i in range(0, len(eval_samples), batch_size):
        batch_samples = eval_samples[i:i + batch_size]
        batch_prompts = [sample["query"] for sample in batch_samples]

        # Use the unified generation path (with chat template) for consistency
        completions, generated_ids, _ = generate_and_decode(
            model,
            tokenizer,
            batch_prompts,
            max_new_tokens,
            enable_thinking=True,
        )

        # Re-tokenize prompts to get prompt lengths for response length calculation
        tokenized_input = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        prompt_lengths = (tokenized_input.input_ids != PAD_TOKEN_ID).sum(dim=1)
        batch_response_lengths = []
        for row_idx in range(generated_ids.size(0)):
            prompt_len = prompt_lengths[row_idx].item()
            response_ids = generated_ids[row_idx, prompt_len:]
            response_len = (response_ids != PAD_TOKEN_ID).sum().item()
            batch_response_lengths.append(response_len)

        # Calculate rewards for the batch
        batch_rewards = math_reward_func(completions, batch_prompts)
        all_rewards.extend(batch_rewards)

        # Accumulate statistics
        total_response_length += float(sum(batch_response_lengths))
        for reward in batch_rewards:
            total_reward += reward
            total_samples += 1
            if reward > 0:
                success_count += 1

    # Calculate metrics
    avg_reward = total_reward / total_samples if total_samples > 0 else 0.0
    reward_std = (
        torch.tensor(all_rewards, dtype=torch.float32).std().item() if total_samples > 0 else 0.0
    )
    success_rate = success_count / total_samples if total_samples > 0 else 0.0
    avg_response_length = total_response_length / total_samples if total_samples > 0 else 0.0

    return {
        "eval_reward_mean": avg_reward,
        "eval_reward_std": reward_std,
        "eval_success_rate": success_rate,
        "eval_avg_response_length": avg_response_length,
        "eval_samples": total_samples,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main():
    # Set print options to print full tensors
    torch.set_printoptions(threshold=10_000, linewidth=200)

    parser = argparse.ArgumentParser(description="GRPO fine-tuning for Qwen3-1.7B on 24-game math problems")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="HuggingFace model identifier.",
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--steps", type=int, default=40, help="Total number of optimization steps.")
    parser.add_argument("--rollouts_per_prompt", type=int, default=8, help="Number of rollouts per prompt.")
    parser.add_argument("--prompts_per_generation", type=int, default=1, help="Number of unique prompts for each generation step.")
    parser.add_argument("--clip_ratio", type=float, default=0.2, help="PPO-style clip ratio")
    parser.add_argument("--kl_coef", type=float, default=0.01, help="KL penalty coefficient")
    parser.add_argument("--max_new_tokens", type=int, default=1200, help="Maximum new tokens to generate")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of prompts to sample from for each optimization step.")
    parser.add_argument("--epochs_per_batch", type=int, default=4, help="Number of optimization epochs to run on each collected batch of experience")
    parser.add_argument("--minibatch_size", type=int, default=1, help="Size of minibatches for optimization.")
    
    # LoRA configuration
    parser.add_argument("--use_lora", action="store_true", default=True, help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")
    
    # Memory optimization
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing to save memory")
    
    # Wandb and evaluation configuration
    parser.add_argument("--use_wandb", action="store_true", default=True, help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="grpo-math-training", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default="custom-grpo", help="W&B run name")
    parser.add_argument("--eval_size", type=int, default=10, help="Number of problems for evaluation")
        
    # KL threshold configuration
    parser.add_argument("--kl_threshold", type=float, default=10, help="KL divergence threshold for early stopping")
    
    # Revision configuration
    parser.add_argument("--use_revision", action="store_true", default=False, help="Use revision model to revise completions during sampling.")
    
    # Learning rate scheduler configuration
    parser.add_argument("--lr_schedule", action="store_true", default=True, help="Use linear learning rate decay")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Minimum learning rate as ratio of initial LR (default: 0.1 = 10% of initial LR)")
    
    # Gradient clipping configuration
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Gradient clipping norm. Set to 0 or negative to disable clipping")

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

    tokenizer.padding_side = "left"

    # Pad token for Qwen3-1.7B is <|endoftext|> = 151643
    # EOS token is <|im_end|> = 151645
    # Therefore this conditional does not execute on Qwen3-1.7B (presumably all Qwen3 models).
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    global PAD_TOKEN_ID
    PAD_TOKEN_ID = tokenizer.pad_token_id
    global EOS_TOKEN_ID
    EOS_TOKEN_ID = tokenizer.eos_token_id

    print(f"Tokenizer pad token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f"Tokenizer EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
    
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)

    # Prefer FlashAttention-2 on capable GPUs (e.g., H100). Fallback to SDPA or omit
    # if the model class does not accept the kwarg.
    attn_impl_kwargs = {}
    try:
        import flash_attn  # type: ignore
        attn_impl_kwargs["attn_implementation"] = "flash_attention_2"
        print("Attention backend: flash_attention_2")
    except Exception as e:
        # If flash-attn is unavailable, prefer SDPA flash kernels when supported
        attn_impl_kwargs["attn_implementation"] = "sdpa"
        print(f"flash-attn not available; using SDPA. Details: {e}")

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            config=config,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            **attn_impl_kwargs,
        )
    except TypeError:
        # Some custom model classes may not accept attn_implementation
        print("Model class does not accept 'attn_implementation'; loading without it.")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            config=config,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
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

    # The user now provides total optimization steps directly.
    total_optim_steps = args.steps

    # Updates per epoch
    updates_per_epoch = args.batch_size / args.minibatch_size
    
    # Calculate the number of data collection cycles.
    collection_steps = math.ceil(total_optim_steps / args.epochs_per_batch / updates_per_epoch)
    print(f"Total optimization steps: {total_optim_steps}")
    print(f"Data collection steps: {collection_steps} (total_optim_steps / epochs_per_batch)")

    trainer = GRPOTrainer(
        model,
        ref_model=ref_model,
        lr=args.lr,
        clip_ratio=args.clip_ratio,
        kl_coef=args.kl_coef,
        total_steps=total_optim_steps,
        lr_schedule=args.lr_schedule,
        min_lr_ratio=args.min_lr_ratio,
        grad_clip_norm=args.grad_clip_norm if args.grad_clip_norm > 0 else None,
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
        collection_steps=collection_steps,
        batch_size=args.batch_size,
        epochs_per_batch=args.epochs_per_batch,
        rollouts_per_prompt=args.rollouts_per_prompt,
        prompts_per_generation=args.prompts_per_generation,
        max_new_tokens=args.max_new_tokens,
        eval_dataset=eval_dataset,
        use_wandb=args.use_wandb,
        kl_threshold=args.kl_threshold,
        use_revision=args.use_revision,
        minibatch_size=args.minibatch_size,
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
