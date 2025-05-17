import pdb
# train_grpo.py
#
# See https://github.com/willccbb/verifiers for ongoing developments
#
import wandb
import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from easy_dataset.easiest_arc_dataset import dataset, grid_to_string
from claude import call_claude_api

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

from score import score_arc_solution

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]

    print(
        '*' * 20,
        f"\nResponse:\n{responses[0]}\n\n", '*' * 20,
        '*' * 20
    )
    return [score_arc_solution(r, a) for r, a in zip(extracted_responses, answer)]


def generate_judge_prompt(context, completion, answer) -> str:
    prompt = "You are a teacher who is guiding a student to improve his reasoning skills.\n\n"
    prompt += "You need to evaluate if the student's reasoning is in the right direction. The student has not arrived to the answer yet. You only need to evaluate current thinking process."
    prompt += "The question is:\n" + context + "\n\n" + "The actual answer is:\n" + grid_to_string(answer) + "\n\n"
    prompt += "The student is thinking:\n" + completion + "\n\n"
    prompt += "Please evaluate and score the student's reasoning on a scale from 0 to 100."
    prompt += "Please provide the score in <score>score</score> tags. No need to reason yourself or justify the score."
    return prompt

def llm_feedback_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    judge_prompts = [generate_judge_prompt(p[0]['content'], c[0]['content'], a) 
                    for p, c, a in zip(prompts, completions, answer)]
    
    scores = []
    # Call Claude API for each prompt and extract scores
    for judge_prompt in judge_prompts:
        # Replace this with your actual Claude API call
        claude_response = call_claude_api(judge_prompt)
        
        # Extract score from Claude's response using regex to find content between <score> tags
        import re
        score_match = re.search(r'<score>(\d+)</score>', claude_response)
        
        if score_match:
            score = float(score_match.group(1))
            scores.append(score)
        else:
            # If no score is found, default to 0
            scores.append(0.0)
    
    return scores


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
    gradient_accumulation_steps=8,
    num_generations=8,
    max_prompt_length=1860,
    max_completion_length=100,
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
        correctness_reward_func,
        llm_feedback_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
    # peft_config=peft_config
)
trainer.train()
