from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model
import wandb
import torch
import random
import re

model_name = "Qwen/Qwen3-1.7B"
dataset_size = 500

def parse_completion(completion: str) -> (str, str):
    """
    Parses a completion string that may contain <think>...</think> blocks.
    Extracts thinking content and the final answer.
    
    Args:
        completion (str): The model's completion string.
        
    Returns:
        tuple[str, str]: A tuple of (thinking_content, final_content).
    """
    end_tag = '</think>'
    start_tag = '<think>'
    
    end_tag_pos = completion.rfind(end_tag)
    
    if end_tag_pos != -1:
        content = completion[end_tag_pos + len(end_tag):].strip()
        
        # Extract thinking content
        # The thinking content is between <think> and </think>
        # We look for the last <think> before the last </think>
        think_part = completion[:end_tag_pos]
        start_tag_pos = think_part.rfind(start_tag)
        
        if start_tag_pos != -1:
            thinking_content = think_part[start_tag_pos + len(start_tag):].strip()
        else:
            # Fallback if no start tag is found before the end tag
            thinking_content = ""
        
        return thinking_content, content
    
    # No </think> tag found
    return "", completion.strip()

def is_correct(content, target):
    """
    Evaluates the mathematical expression in content and returns True if it equals the target value.
    
    Args:
        content (str): The mathematical expression to evaluate
        target (float): The target value the expression should equal
        
    Returns:
        bool: True if the expression evaluates to the target value, False otherwise
    """
    try:
        # Clean the content by stripping whitespace and newlines
        expression = content.strip().replace('×', '*').replace('÷', '/')
        
        # Evaluate the mathematical expression
        result = eval(expression)
        
        # Check if the result equals the target (with some floating point tolerance)
        return abs(result - target) < 1e-10
        
    except (SyntaxError, NameError, ZeroDivisionError, TypeError, ValueError) as e:
        # Return False if the expression is invalid or causes an error
        # print(f"Error evaluating expression '{content}': {e}")
        return False

def decompose_target(target, num_count=4, num_range=10):
    """
    Decomposes a target number into num_count numbers using +, -, *, / operators.
    
    Args:
        target (float): The target number to decompose
        num_count (int): Number of numbers to generate (default 4)
        num_range (int): Range for random numbers (1 to num_range)
        
    Returns:
        tuple: (list of numbers, expression string) that can be combined with operators to equal target
    """
    if num_count == 1:
        if abs(target - round(target)) < 1e-10:
            return [round(target)], str(round(target))
        return [target], str(target)
    
    # Pick a random number and operator
    rand_num = random.randint(1, num_range)
    operator = random.choice(['+', '-', '*', '/'])
    
    # Calculate what the previous result needs to be based on inverse operation
    try:
        if operator == '+':
            # If we want prev_result + rand_num = target, then prev_result = target - rand_num
            prev_result = target - rand_num
        elif operator == '-':
            # If we want prev_result - rand_num = target, then prev_result = target + rand_num
            prev_result = target + rand_num
        elif operator == '*':
            # If we want prev_result * rand_num = target, then prev_result = target / rand_num
            if rand_num == 0:
                # Avoid division by zero
                return decompose_target(target, num_count, num_range)
            prev_result = target / rand_num
        elif operator == '/':
            # If we want prev_result / rand_num = target, then prev_result = target * rand_num
            prev_result = target * rand_num
            
        # Recurse to get the remaining numbers and expression
        remaining_numbers, prev_expression = decompose_target(prev_result, num_count - 1, num_range)
        
        # Build the complete expression
        # Add parentheses for clarity when needed
        if num_count > 2 and operator in ['*', '/'] and any(op in prev_expression for op in ['+', '-']):
            expression = f"({prev_expression}) {operator} {rand_num}"
        else:
            expression = f"{prev_expression} {operator} {rand_num}"
        
        # Add our random number to the list
        remaining_numbers.append(rand_num)
        return remaining_numbers, expression
        
    except (ZeroDivisionError, OverflowError):
        # If we get an invalid operation, try again
        return decompose_target(target, num_count, num_range)

def generate_problem(target, num_count=4, num_range=10):
    """
    Generates a problem by sampling from decompose_target until all numbers are integers within 1..num_range.
    
    Args:
        target (float): The target number to decompose
        num_count (int): Number of numbers to generate (default 4)
        num_range (int): Range for random numbers (1 to num_range)
        
    Returns:
        tuple: (list of numbers, expression string) where all numbers are integers in [1, num_range]
    """
    max_attempts = 1000  # Prevent infinite loops
    
    for attempt in range(max_attempts):
        numbers, expression = decompose_target(target, num_count, num_range)
        
        # Check if all numbers are integers and within range
        if (numbers and 
            all(isinstance(num, int) or (isinstance(num, float) and num.is_integer()) for num in numbers) and
            all(1 <= int(num) <= num_range for num in numbers)):
            
            # Convert all numbers to integers for clean output
            int_numbers = [int(num) for num in numbers]
            random.shuffle(int_numbers)
            return int_numbers, expression
    
    # If we couldn't find a valid solution after max_attempts, return None
    print(f"Warning: Could not generate valid problem for target {target} after {max_attempts} attempts")
    return None, None

def test_generate():
    # Test the generate_problem function
    print("\n" + "="*50)
    print("Testing generate_problem function:")
    print("="*50)

    # Test with different targets
    targets = [24, 24, 24, 24]

    for target in targets:
        print(f"\nTarget: {target}")
        
        # Generate 3 examples for each target using the improved function
        for i in range(3):
            numbers, expression = generate_problem(target, num_count=4, num_range=10)
            
            print(f"  Attempt {i+1}:")
            if numbers and expression:
                print(f"    Numbers: {numbers}")
                print(f"    ✓ Expression: {expression} = {target}")
                print(f"    ✓ Is correct: {is_correct(expression, target)}")
            else:
                print(f"    ✗ Could not find valid expression after 100 attempts")
                
        print("-" * 30)

def test_inference():
    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    # prepare the model input
    prompt = "Using the numbers 10, 5, 4, 4 exactly once in mathematical notation using addition, subtraction, multiplication, division, and parentheses, create an expression that equals 24. Answer exactly in plain mathematical notation (DO NOT USE LATEX), WITH NO ADDITIONAL TEXT. For example, if the provided numbers are 3, 3, 2, 8, a valid answer would be: (3 / 3 + 2) * 8."
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("starting inference...")

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768,
        temperature=0.7,
        do_sample=True,
        repetition_penalty=1.1
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    completion = tokenizer.decode(output_ids, skip_special_tokens=True)
    thinking_content, content = parse_completion(completion)

    print("thinking content:", thinking_content)
    print("content:", content)

    # Check if the generated expression is correct
    result_is_correct = is_correct(content, 24)
    print(f"Is the answer correct? {result_is_correct}")

def generate_math_problems(tokenizer):
    """
    Generator function that creates math problems using generate_problem.
    Yields dictionary with 'prompt' containing the problem description, formatted with thinking template.
    """
    targets = [16]  # Various target numbers
    
    for _ in range(dataset_size):
        target = random.choice(targets)
        numbers, expression = generate_problem(target, num_count=3, num_range=10)
        
        if numbers and expression:  # Only yield if we successfully generated a problem
            # Create prompt similar to test_inference
            numbers_str = ", ".join(map(str, numbers))
            # prompt_content = f"Using the numbers {numbers_str} exactly once in mathematical notation using addition, subtraction, multiplication, division, and/or parentheses, create an expression that equals {target}. Answer exactly in plain mathematical notation (DO NOT USE LATEX), WITH NO ADDITIONAL TEXT. For example, if the provided numbers are 3, 3, 2, 8, a valid answer would be: (3 / 3 + 2) * 8. As soon as you find a correct expression, output your answer."
            prompt_content = f"Using the numbers {numbers_str} exactly once in mathematical notation using addition, subtraction, multiplication, division, and/or parentheses, create an expression that equals {target}. Keep your reasoning in the <think> block brief. Answer exactly in plain mathematical notation (DO NOT USE LATEX), WITH NO ADDITIONAL TEXT. For example, if the provided numbers are 2, 4, 4, a valid answer would be: (4 + 4) * 2. Or, if the numbers were 8, 1, 7, a valid answer would be 8 + 1 + 7. ANSWER AS SOON AS A CORRECT EXPRESSION IS FOUND."
            
            messages = [{"role": "user", "content": prompt_content}]
            
            # Apply chat template with thinking mode enabled
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            
            yield {
                "prompt": prompt,
                "target": target,
                "numbers": numbers
            }

def math_reward_func(completions, prompts, **kwargs):
    """
    Reward function that evaluates mathematical correctness using is_correct.
    
    Args:
        completions: List of generated completions
        prompts: List of prompts (to extract target values)
        **kwargs: Additional arguments
        
    Returns:
        List of reward scores (1.0 for correct, 0.0 for incorrect)
    """
    rewards = []
    
    for completion, prompt in zip(completions, prompts):
        # Extract target from prompt
        # Look for "equals X" pattern in the prompt
        target_match = re.search(r'equals (\d+)', prompt)
        
        # Clean the completion and check if it's correct
        _, content = parse_completion(completion)

        if target_match:
            target = float(target_match.group(1))
            is_correct_answer = is_correct(content, target)
            reward = 1.0 if is_correct_answer else 0.0
        else:
            # If we can't extract target, give low reward
            reward = 0.0
        
        # Penalize completions that are clearly unfinished (unclosed <think> tag)
        if '<think>' in completion and '</think>' not in completion:
            reward = -1.0
            
        print("\n-----")
        print(f"Prompt: {prompt}")
        print(f"Completion: {completion}")
        print(f"Parsed Content: {content}")
        print(f"Reward: {reward}")
        print("-----")
            
        rewards.append(reward)
    
    return rewards

def test_reward():
    """
    Test the reward function with some sample completions and prompts
    """
    test_completions = [
        "(4 + 4) * 3",  # Should be correct for target 24
        "4 + 4 + 3",    # Should be incorrect for target 24
        "10 * 2asdf"    # Should be correct for target 24
    ]
    
    test_prompts = [
        "Using the numbers 4, 4, 3, 3 exactly once in mathematical notation using addition, subtraction, multiplication, division, and parentheses, create an expression that equals 24.",
        "Using the numbers 4, 4, 3, 3 exactly once in mathematical notation using addition, subtraction, multiplication, division, and parentheses, create an expression that equals 24.",
        "Using the numbers 10, 2, 4, 4 exactly once in mathematical notation using addition, subtraction, multiplication, division, and parentheses, create an expression that equals 24."
    ]
    
    rewards = math_reward_func(test_completions, test_prompts)
    
    print("\nTesting reward function:")
    for i, (completion, prompt, reward) in enumerate(zip(test_completions, test_prompts, rewards)):
        print(f"\nTest {i+1}:")
        print(f"  Completion: {completion}")
        print(f"  Reward: {reward}")

def get_training_config():
    """
    Get training configuration parameters.
    You can modify these values to experiment with different settings.
    """
    return {
        "model_name": model_name,
        "learning_rate": 5e-5,  # Slightly lower learning rate for more stable training
        "batch_size": 8,  # Increased batch size for A100. Reduce if you encounter OOM errors.
        "gradient_accumulation_steps": 2,  # Effective batch size of 16
        "num_epochs": 1,
        "max_completion_length": 1500,
        "temperature": 0.7,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "warmup_steps": 5,  # ~10% of total steps
        "logging_steps": 5,
        "save_steps": 5,   # Save every epoch
        "eval_steps": 5,   # Eval every epoch
        "dataset_size": dataset_size,  # Increased dataset size for more examples
    }

def train():
    """
    Train the model using GRPO with math problems generated from generate_problem function.
    """
    config = get_training_config()
    
    # Initialize wandb
    wandb.init(
        project="math-grpo-training",
        name="qwen-math-24-game",
        config=config
    )
    
    # Load tokenizer first to format prompts
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create separate train and eval datasets
    full_dataset = Dataset.from_generator(
        generate_math_problems, gen_kwargs={"tokenizer": tokenizer}
    )
    dataset_size = min(len(full_dataset), wandb.config.dataset_size)

    # Split into train/eval (e.g., 80/20 split)
    train_size = int(0.8 * dataset_size)
    train_dataset = full_dataset.select(range(train_size))
    eval_dataset = full_dataset.select(range(train_size, dataset_size))

    print(f"Created dataset with {len(train_dataset)} training examples and {len(eval_dataset)} evaluation examples")
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=wandb.config.lora_r,
        lora_alpha=wandb.config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=wandb.config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Configure GRPO training arguments
    grpo_config = GRPOConfig(
        output_dir="./grpo-math-model",
        learning_rate=wandb.config.learning_rate,
        per_device_train_batch_size=wandb.config.batch_size,
        gradient_accumulation_steps=wandb.config.gradient_accumulation_steps,
        num_train_epochs=wandb.config.num_epochs,
        max_completion_length=wandb.config.max_completion_length,
        temperature=wandb.config.temperature,
        logging_steps=wandb.config.logging_steps,
        save_steps=wandb.config.save_steps,
        eval_steps=wandb.config.eval_steps,
        save_total_limit=3,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="wandb",
        run_name="qwen-math-grpo",
        gradient_checkpointing=True,
        dataloader_drop_last=True,
        eval_strategy="steps",
        save_strategy="steps",
        warmup_steps=wandb.config.warmup_steps,
        bf16=True,
    )

    # Initialize trainer with math reward function
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        processing_class=tokenizer,
        reward_funcs=[math_reward_func],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Starting GRPO training...")
    trainer.train()
    
    # Save the final model
    trainer.save_model("./grpo-math-final")
    
    print("Training completed!")
    wandb.finish()

def test_training_setup():
    """
    Test the training setup without actually training to make sure everything works.
    """
    print("Testing training setup...")
    
    try:
        # Test imports
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOTrainer, GRPOConfig
        from peft import LoraConfig, get_peft_model
        import wandb
        print("✓ All imports successful")
        
        # Test model loading
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✓ Tokenizer loaded successfully")
        
        # Test dataset generation
        dataset = Dataset.from_generator(
            generate_math_problems,
            gen_kwargs={"tokenizer": tokenizer},
        )
        dataset = dataset.select(range(10))  # Just test with 10 examples
        print(f"✓ Dataset created with {len(dataset)} examples")
        
        # Test reward function
        test_completions = ["4 + 4 + 4 * 4", "2 * 12"]
        test_prompts = [
            "Using the numbers 4, 4, 4, 4 exactly once in mathematical notation using addition, subtraction, multiplication, division, and parentheses, create an expression that equals 24.",
            "Using the numbers 2, 12, 1, 1 exactly once in mathematical notation using addition, subtraction, multiplication, division, and parentheses, create an expression that equals 24."
        ]
        rewards = math_reward_func(test_completions, test_prompts)
        print(f"✓ Reward function working, rewards: {rewards}")
        
        print("✓ All setup tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Setup test failed: {e}")
        return False

def main():
    train()
    # test_training_setup()
    # test_inference()

if __name__ == "__main__":
    main()