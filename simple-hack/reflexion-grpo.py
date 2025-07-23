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
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the policy gradient loss component, ignoring padded tokens."""
        ratio = torch.exp(new_logp - old_logp)
        if self.clip_ratio > 0:
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            pg_loss1 = -(ratio * advantages)
            pg_loss2 = -(clipped_ratio * advantages)
            pg_losses = torch.max(pg_loss1, pg_loss2)
            
            # Debug: policy gradient loss statistics before mean, using mask
            print(f"[DEBUG PG] pg_loss1: mean={pg_loss1[mask].mean().item():.4f}, std={pg_loss1[mask].std().item():.4f}, min={pg_loss1[mask].min().item():.4f}, max={pg_loss1[mask].max().item():.4f}")
            print(f"[DEBUG PG] pg_loss2: mean={pg_loss2[mask].mean().item():.4f}, std={pg_loss2[mask].std().item():.4f}, min={pg_loss2[mask].min().item():.4f}, max={pg_loss2[mask].max().item():.4f}")
            print(f"[DEBUG PG] final_pg_losses: mean={pg_losses[mask].mean().item():.4f}, std={pg_losses[mask].std().item():.4f}, min={pg_losses[mask].min().item():.4f}, max={pg_losses[mask].max().item():.4f}")
            
            # Count clipping statistics, using mask
            clipped_mask = (ratio < 1.0 - self.clip_ratio) | (ratio > 1.0 + self.clip_ratio)
            clipped_fraction = clipped_mask[mask].float().mean().item()
            print(f"[DEBUG PG] clipped_fraction: {clipped_fraction:.3f}")
            
            return pg_losses[mask].mean()
        else:
            unclipped_losses = -(ratio * advantages)
            return unclipped_losses[mask].mean()

    def _kl_loss(
        self,
        new_logp: torch.Tensor,
        ref_logp: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the KL divergence loss against the reference model, ignoring padded tokens."""
        log_ratio_ref = ref_logp - new_logp
        ratio_ref = torch.exp(log_ratio_ref)
        kl_losses = ratio_ref - log_ratio_ref - 1
        
        # Debug: KL loss statistics, using mask
        print(f"[DEBUG KL] log_ratio_ref: mean={log_ratio_ref[mask].mean().item():.4f}, std={log_ratio_ref[mask].std().item():.4f}, min={log_ratio_ref[mask].min().item():.4f}, max={log_ratio_ref[mask].max().item():.4f}")
        print(f"[DEBUG KL] kl_losses: mean={kl_losses[mask].mean().item():.4f}, std={kl_losses[mask].std().item():.4f}, min={kl_losses[mask].min().item():.4f}, max={kl_losses[mask].max().item():.4f}")
        
        return kl_losses[mask].mean()

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        prompt_length: int,
        pad_token_id: int,
        old_logp: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Compute the loss for a batch, but do not perform an optimization step.
        This is used for gradient accumulation.

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

        # Create mask to ignore padding tokens in loss calculations.
        # The mask should include all real tokens and the first pad/EOS token,
        # but exclude all subsequent padding tokens.
        is_pad = (target_actions == pad_token_id)
        # The cumulative sum will be 0 for real tokens, 1 for the first pad
        # token, and >1 for subsequent pad tokens. We keep everything <= 1.
        mask = torch.cumsum(is_pad.to(torch.int), dim=1) <= 1

        # Compute loss components
        pg_loss = self._pg_loss(new_logp, old_logp, advantages, mask)
        kl_loss = self._kl_loss(new_logp, ref_logp, mask)
        loss = pg_loss + self.kl_coef * kl_loss

        return {
            "loss": loss,  # Return the loss tensor for backward pass
            "pg_loss": pg_loss.item(),
            "kl_loss": kl_loss.item(),
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
        gradient_accumulation_steps: int,
        optim_epochs: int,
        batch_size: int,
        max_new_tokens: int,
        eval_dataset: Optional[Any] = None,
        use_wandb: bool = False,
        kl_threshold: float = 0.02,
        use_revision: bool = False,
    ) -> Dict[str, Any]:
        """Train the model using GRPO.
        
        Parameters
        ----------
        tokenizer : tokenizer for the model
        steps : total number of optimization steps
        gradient_accumulation_steps : number of batches to accumulate gradients over
        optim_epochs : number of optimization epochs to run on each collected batch of experience
        batch_size : batch size for training
        max_new_tokens : maximum new tokens to generate
        eval_dataset : optional dataset for evaluation
        use_wandb : whether to log to wandb
        kl_threshold : KL divergence threshold for early stopping
        use_revision : whether to use revision model for sampling
        
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
        # Set the model back to train mode for training
        self.model.train()
        
        # Main training loop (steps are now data collection cycles)
        for step in range(1, steps + 1):
            
            # --- 1. Data Collection Phase ---
            experience_buffer = []
            step_rewards_mean = []
            step_rewards_max = []
            step_success_rates = []

            print(f"\nCollecting experience for step {step}/{steps}...")
            for micro_step in range(gradient_accumulation_steps):
                # Sample a fresh batch for each accumulation step
                if use_revision:
                    batch = sample_and_revise_math_batch(
                        model=self.model,
                        tokenizer=tokenizer,
                        revision_model=self.model,
                        batch_size=batch_size,
                        max_new_tokens=max_new_tokens,
                        pad=True,
                        disable_adapter=False,
                        enable_thinking=False
                    )
                else:
                    batch = sample_math_batch(self.model, tokenizer, batch_size=batch_size, max_new_tokens=max_new_tokens)
                
                input_ids, actions, rewards, prompt_length, pad_token_id, _, _ = batch
                
                # Pre-compute old_logp based on the policy at the time of collection
                with torch.no_grad():
                    outputs = self.model(input_ids.to(self.device))
                    logits = outputs.logits[:, prompt_length-1:-1, :]
                    target_actions = actions.to(self.device)
                    old_logp = self._old_log_probs(logits, target_actions)
                
                experience_buffer.append({
                    'input_ids': input_ids, 'actions': actions, 'rewards': rewards,
                    'prompt_length': prompt_length, 'pad_token_id': pad_token_id,
                    'old_logp': old_logp
                })
                
                # Track and log rewards from this collection micro-batch
                batch_reward_mean = rewards.mean().item()
                training_rewards.extend(rewards.tolist())
                step_rewards_mean.append(batch_reward_mean)
                step_rewards_max.append(rewards.max().item())
                step_success_rates.append((rewards > 0).float().mean().item())
                print(f"  Collected micro-batch {micro_step+1}/{gradient_accumulation_steps} | reward: {batch_reward_mean:.3f}")
            
            # --- 2. Optimization Phase ---
            for epoch in range(optim_epochs):
                self.optimizer.zero_grad()
                
                epoch_losses = []
                epoch_pg_losses = []
                epoch_kls = []

                # Iterate over the collected experience
                for micro_batch_data in experience_buffer:
                    # Compute loss for the micro-batch using the pre-computed old_logp
                    metrics = self.compute_loss(
                        input_ids=micro_batch_data['input_ids'],
                        actions=micro_batch_data['actions'],
                        rewards=micro_batch_data['rewards'],
                        prompt_length=micro_batch_data['prompt_length'],
                        pad_token_id=micro_batch_data['pad_token_id'],
                        old_logp=micro_batch_data['old_logp']
                    )
                    loss = metrics['loss']
                    
                    # Normalize loss for accumulation across the buffer
                    loss = loss / len(experience_buffer)
                    
                    # Accumulate gradients
                    loss.backward()

                    # Store metrics for logging
                    epoch_losses.append(loss.item() * len(experience_buffer))
                    epoch_pg_losses.append(metrics['pg_loss'])
                    epoch_kls.append(metrics['kl_loss'])
                
                # Clip gradients and perform optimizer step after accumulating over the whole buffer
                total_optim_steps += 1
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                # Step the learning rate scheduler
                if self.scheduler is not None:
                    self.scheduler.step()
                    
                # Log aggregated metrics for the optimization step
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                avg_pg_loss = sum(epoch_pg_losses) / len(epoch_pg_losses)
                avg_kl = sum(epoch_kls) / len(epoch_kls)
                avg_reward_mean = sum(step_rewards_mean) / len(step_rewards_mean)
                avg_reward_max = sum(step_rewards_max) / len(step_rewards_max)
                avg_success_rate = sum(step_success_rates) / len(step_success_rates)
                
                current_lr = self.optimizer.param_groups[0]['lr']

                if use_wandb:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/pg_loss": avg_pg_loss,
                        "train/kl_divergence": avg_kl,
                        "train/learning_rate": current_lr,
                        "train/batch_reward_mean": avg_reward_mean,
                        "train/batch_reward_max": avg_reward_max,
                        "train/batch_success_rate": avg_success_rate,
                        "step": total_optim_steps,
                        "collection_step": step,
                        "optim_epoch": epoch + 1,
                    })
                
                print(
                    f"Optim Step {total_optim_steps:05d} | Collection Step {step}/{steps}, Epoch {epoch+1}/{optim_epochs} | "
                    f"loss: {avg_loss:.4f} | "
                    f"kl: {avg_kl:.4f} | "
                    f"lr: {current_lr:.2e} | "
                    f"reward: {avg_reward_mean:.3f} | "
                    f"success: {avg_success_rate:.1%}"
                )
                    
                # Early stopping if KL divergence gets too high
                if avg_kl > kl_threshold:
                    print(f"  Early stopping epoch due to high KL divergence: {avg_kl:.4f}")
                    break
            
            # Break outer loop if KL is high
            if avg_kl > kl_threshold:
                print(f"  Early stopping collection step due to high KL divergence.")
                break

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
    with torch.no_grad():
        generated_ids = model.generate(**kwargs)

    # Restore original states
    model.config.use_cache = original_use_cache
    if was_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return generated_ids


def generate_and_decode(model, tokenizer, prompts, max_new_tokens, disable_adapter=False, enable_thinking: bool = True, **gen_kwargs):
    """Generates completions from a model and decodes them."""
    
    # Convert prompts to chat message format
    messages_list = [[{"role": "user", "content": p}] for p in prompts]
    
    # Apply chat template with thinking mode control
    # This creates the full prompt string including special tokens
    processed_prompts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        ) for messages in messages_list
    ]

    tokenized = tokenizer(processed_prompts, return_tensors="pt", padding=True, truncation=True)
    
    # Base generation arguments
    base_gen_kwargs = {
        "input_ids": tokenized["input_ids"].to(model.device),
        "attention_mask": tokenized["attention_mask"].to(model.device),
        "max_new_tokens": max_new_tokens,
        "temperature": 0.7,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
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
        
    # Extract, decode, and return completions
    prompt_length = tokenized["input_ids"].shape[1]
    generated_tokens = generated_ids[:, prompt_length:]
    completions = [tokenizer.decode(gen, skip_special_tokens=True) for gen in generated_tokens]
    
    return completions


def _create_batch_from_prompts(prompts, completions, tokenizer, batch_size, pad):
    """Create a batch for the trainer from prompts and completions."""
    rewards = torch.tensor(math_reward_func(completions, prompts), dtype=torch.float32)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    
    # Get a consistent prompt length by padding
    tokenized_prompts = tokenizer(prompts, padding=True, return_tensors="pt")
    prompt_length = tokenized_prompts['input_ids'].shape[1]

    # Tokenize completions and create full sequences
    full_sequences = []
    actions = []
    
    tokenized_prompts_list = tokenizer(prompts, padding=False, truncation=True)['input_ids']
    tokenized_completions_list = tokenizer(completions, padding=False, truncation=True)['input_ids']

    for i in range(batch_size):
        prompt_toks = torch.tensor(tokenized_prompts_list[i], dtype=torch.long)
        completion_toks = torch.tensor(tokenized_completions_list[i], dtype=torch.long)
        
        full_seq = torch.cat([prompt_toks, completion_toks])
        full_sequences.append(full_seq)
        actions.append(completion_toks)

    if pad:
        input_ids, actions = pad_sequences_for_batch(full_sequences, actions, batch_size, pad_token_id)
    else:
        input_ids = full_sequences
        
    return input_ids, actions, rewards, prompt_length, pad_token_id

def pad_sequences_for_batch(full_sequences, generated_tokens, batch_size, pad_token_id):
    """Pad sequences to same length for batch processing.
    
    Parameters
    ----------
    full_sequences : list of torch.Tensor
        Full sequences (prompt + generated tokens)
    generated_tokens : list of torch.Tensor
        Generated tokens only
    batch_size : int
        Size of the batch
    pad_token_id : int
        Token ID to use for padding
        
    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Padded input_ids and actions tensors
    """
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
    
    return input_ids, actions

def sample_and_revise_math_batch(
    model, 
    tokenizer, 
    revision_model,
    batch_size: int, 
    max_new_tokens: int, 
    pad: bool, 
    disable_adapter: bool, 
    enable_thinking: bool
):
    """
    Samples a batch, revises it, and returns the revised batch.
    1. Sample a batch of completions from the base model.
    2. Use a revision model to revise the completions.
    3. Return the revised completions and their rewards.
    """
    # 1. First pass: Sample from the base model to get initial solutions
    _, _, _, _, _, prompts, initial_completions = sample_math_batch(
        model, tokenizer, batch_size, max_new_tokens, pad=True
    )

    # 2. Second pass: Construct revision prompts and revise with the revision_model
    revision_prompts = []
    for i in range(batch_size):
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
    revised_completions = generate_and_decode(
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
    input_ids, actions, rewards, prompt_length, pad_token_id = _create_batch_from_prompts(
        prompts, final_completions, tokenizer, batch_size, pad
    )
    
    return input_ids, actions, rewards, prompt_length, pad_token_id, prompts, final_completions


def sample_math_batch(model, tokenizer, batch_size: int = 4, max_new_tokens: int = 512, pad: bool = True):
    """Generate a batch of math problems and model completions for GRPO training."""
    
    # Generate one math problem and use it for all batch elements
    problem_generator = generate_math_problems(tokenizer, 1)
    single_problem = next(problem_generator)
    problems = [single_problem for _ in range(batch_size)]
    
    # Extract prompts (all the same now)
    prompts = [problem["prompt"] for problem in problems]

    # Generate completions using the model
    completions = generate_and_decode(model, tokenizer, prompts, max_new_tokens, enable_thinking=True)
    
    # Create the batch from prompts and generated completions
    input_ids, actions, rewards, prompt_length, pad_token_id = _create_batch_from_prompts(
        prompts, completions, tokenizer, batch_size, pad
    )
    
    return input_ids, actions, rewards, prompt_length, pad_token_id, prompts, completions


def evaluate_model(model, tokenizer, eval_dataset, max_new_tokens=512, batch_size=12):
    """Batched evaluation function that generates completions and calculates rewards."""
    
    model.eval()
    total_reward = 0.0
    total_samples = 0
    success_count = 0
    
    # Convert dataset to list if it's not already
    eval_samples = list(eval_dataset)
    
    # Process evaluation dataset in batches
    for i in range(0, len(eval_samples), batch_size):
        batch_samples = eval_samples[i:i+batch_size]
        batch_prompts = [sample["query"] for sample in batch_samples]
        
        # Tokenize the batch of prompts
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        
        # Generate completions for the batch
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
        
        # Extract only the generated parts and decode completions
        prompt_length = inputs["input_ids"].shape[1]
        generated_tokens = generated[:, prompt_length:]
        completions = [tokenizer.decode(gen, skip_special_tokens=True) for gen in generated_tokens]
        
        # Calculate rewards for the batch
        batch_rewards = math_reward_func(completions, batch_prompts)
        
        # Accumulate statistics
        for reward in batch_rewards:
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
    parser.add_argument("--steps", type=int, default=40, help="Total number of optimization steps.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (can be increased with LoRA)")
    parser.add_argument("--clip_ratio", type=float, default=0.2, help="PPO-style clip ratio")
    parser.add_argument("--kl_coef", type=float, default=0.01, help="KL penalty coefficient")
    parser.add_argument("--max_new_tokens", type=int, default=1200, help="Maximum new tokens to generate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of batches to accumulate gradients over before performing an optimizer step")
    parser.add_argument("--optim_epochs", type=int, default=4, help="Number of optimization epochs to run on each collected batch of experience")
    
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
    
    # Revision configuration
    parser.add_argument("--use_revision", action="store_true", default=False, help="Use revision model to revise completions during sampling.")
    
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

    # The user now provides total optimization steps directly.
    total_optim_steps = args.steps
    
    # Calculate the number of data collection cycles.
    collection_steps = math.ceil(total_optim_steps / args.optim_epochs)
    print(f"Total optimization steps: {total_optim_steps}")
    print(f"Data collection steps: {collection_steps} (total_optim_steps / optim_epochs)")

    trainer = GRPOTrainer(
        model,
        ref_model=ref_model,
        lr=args.lr,
        clip_ratio=args.clip_ratio,
        kl_coef=args.kl_coef,
        dr=args.dr,
        total_steps=total_optim_steps,
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
        steps=collection_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim_epochs=args.optim_epochs,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        eval_dataset=eval_dataset,
        use_wandb=args.use_wandb,
        kl_threshold=args.kl_threshold,
        use_revision=args.use_revision,
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
