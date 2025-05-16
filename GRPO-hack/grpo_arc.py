import pdb
# train_grpo.py
#
# See https://github.com/willccbb/verifiers for ongoing developments
#
import wandb
import re
import torch
from datasets import load_dataset, Dataset
from easy_dataset.easiest_arc_dataset import dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer


wandb.init(project="grpo-arc")

def calculate_nonzero_match_percentage(answer_str, output_str):
    """
    Calculate the percentage of matching non-zero characters between answer and output strings.
    Ignores all zeros and only considers matching of non-zero characters.
    
    Args:
        answer_str (str): The reference answer string
        output_str (str): The output string to compare against the answer
        
    Returns:
        float: Percentage of matching non-zero characters (0-100)
    """
    # Remove any whitespace to focus only on actual characters
    answer_clean = ''.join(answer_str.split())
    output_clean = ''.join(output_str.split())
    
    # If either string is empty, return 0%
    if not answer_clean or not output_clean:
        return 0.0
    
    # Get lengths of both strings
    answer_len = len(answer_clean)
    output_len = len(output_clean)
    
    # Count total non-zero characters in answer
    total_nonzero_in_answer = sum(1 for char in answer_clean if char != '0')
    
    # If there are no non-zero characters in answer, return 100% if output has none, 0% otherwise
    if total_nonzero_in_answer == 0:
        return 100.0 if all(char == '0' for char in output_clean) else 0.0
    
    # Count correct non-zero matches
    correct_nonzero_matches = 0
    min_len = min(answer_len, output_len)
    
    for i in range(min_len):
        # Only count matches where the answer has a non-zero character
        if answer_clean[i] != '0' and answer_clean[i] == output_clean[i]:
            correct_nonzero_matches += 1
    
    # Calculate percentage based on the total non-zero characters in answer
    percentage = (correct_nonzero_matches / total_nonzero_in_answer)
    
    return percentage

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    import pdb
    pdb.set_trace()
    print(
        '*' * 20,
        f"\nResponse:\n{responses[0]}\n\n", '*' * 20,
        f"\nExtracted:\n{extracted_responses[0].strip()}",
        f"\nAnswer:\n{answer[0].strip()}\n",
        '\n',
        '*' * 20,
        calculate_nonzero_match_percentage(extracted_responses[0].strip(), answer[0].strip()),
        '\n',
        '*' * 20
    )
    return [2.0 * calculate_nonzero_match_percentage(r.strip(), a.strip()) for r, a in zip(extracted_responses, answer)]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

training_args = GRPOConfig(
    output_dir="./",
    run_name="test-run-arc",
    learning_rate=5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_generations=16,
    max_prompt_length=1860,
    max_completion_length=8000,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    report_to="wandb",
    log_on_each_node=False,
    top_k=50,
    top_p=0.65,
    temperature=0.9,
)

peft_config = LoraConfig(
    r=4,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
)

BASE_MODEL_NAME = "Qwen/Qwen3-0.6B"

# load the tokenizer and the model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.pad_token = '<|endoftext|>'
tokenizer.eos_token_id = 151643

# use peft at your own risk; not working for me with multi-GPU training
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        soft_format_reward_func,
        correctness_reward_func
    ],
    args=training_args,
    train_dataset=dataset,
    # peft_config=peft_config
)
trainer.train()
