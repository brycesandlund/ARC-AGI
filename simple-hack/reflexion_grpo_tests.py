import torch

def test_sample(input_ids, rewards, advantages, loss_mask, prompts, completions, rollouts_per_prompt, prompts_per_generation):
    """
    A simple test function to check the shapes and contents of the sample function's return values.
    """
    print("\\n--- Running test_sample ---")

    # Calculate expected batch size
    expected_batch_size = prompts_per_generation * rollouts_per_prompt

    # 1. Test input_ids
    assert isinstance(input_ids, torch.Tensor), f"input_ids should be a torch.Tensor, but got {type(input_ids)}"
    assert input_ids.dim() == 2, f"input_ids should be a 2D tensor, but has {input_ids.dim()} dimensions"
    assert input_ids.shape[0] == expected_batch_size, f"input_ids.shape[0] should be {expected_batch_size}, but is {input_ids.shape[0]}"
    print(f"✅ input_ids shape is correct: {input_ids.shape}")

    # 2. Test rewards
    assert isinstance(rewards, torch.Tensor), f"rewards should be a torch.Tensor, but got {type(rewards)}"
    assert rewards.dim() == 1, f"rewards should be a 1D tensor, but has {rewards.dim()} dimensions"
    assert rewards.shape[0] == expected_batch_size, f"rewards.shape[0] should be {expected_batch_size}, but is {rewards.shape[0]}"
    print(f"✅ rewards shape is correct: {rewards.shape}")

    # 3. Test advantages
    assert isinstance(advantages, torch.Tensor), f"advantages should be a torch.Tensor, but got {type(advantages)}"
    assert advantages.dim() == 1, f"advantages should be a 1D tensor, but has {advantages.dim()} dimensions"
    assert advantages.shape[0] == expected_batch_size, f"advantages.shape[0] should be {expected_batch_size}, but is {advantages.shape[0]}"
    print(f"✅ advantages shape is correct: {advantages.shape}")

    # 4. Test loss_mask
    assert isinstance(loss_mask, torch.Tensor), f"loss_mask should be a torch.Tensor, but got {type(loss_mask)}"
    assert loss_mask.dim() == 2, f"loss_mask should be a 2D tensor, but has {loss_mask.dim()} dimensions"
    assert loss_mask.shape[0] == expected_batch_size, f"loss_mask.shape[0] should be {expected_batch_size}, but is {loss_mask.shape[0]}"
    assert loss_mask.shape[1] == input_ids.shape[1] - 1, f"loss_mask.shape[1] should be {input_ids.shape[1] - 1}, but is {loss_mask.shape[1]}"
    assert loss_mask.dtype == torch.bool, f"loss_mask should be a boolean tensor, but is {loss_mask.dtype}"
    print(f"✅ loss_mask shape and dtype are correct: {loss_mask.shape}, {loss_mask.dtype}")

    # 5. Test prompts
    assert isinstance(prompts, list), f"prompts should be a list, but got {type(prompts)}"
    assert len(prompts) == expected_batch_size, f"len(prompts) should be {expected_batch_size}, but is {len(prompts)}"
    assert all(isinstance(p, str) for p in prompts), "All elements in prompts should be strings"
    print(f"✅ prompts is a list of strings of correct length: {len(prompts)}")

    # 6. Test completions
    assert isinstance(completions, list), f"completions should be a list, but got {type(completions)}"
    assert len(completions) == expected_batch_size, f"len(completions) should be {expected_batch_size}, but is {len(completions)}"
    assert all(isinstance(c, str) for c in completions), "All elements in completions should be strings"
    print(f"✅ completions is a list of strings of correct length: {len(completions)}")

    print("--- test_sample PASSED ---\\n")
