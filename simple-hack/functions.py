from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import random
import re

def parse_completion(completion: str) -> tuple[str, str]:
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
    return "", ""

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
        expression = content.strip().replace('×', '*').replace('÷', '/').replace('−', '-')
        
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

def test_inference(model_name):
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

def generate_math_problems(tokenizer, dataset_size):
    """
    Generator function that creates math problems using generate_problem.
    Yields dictionary with 'prompt' containing the problem description, formatted with thinking template.
    """
    targets = [24]  # Various target numbers
    
    for _ in range(dataset_size):
        target = random.choice(targets)
        numbers, expression = generate_problem(target, num_count=4, num_range=10)
        
        if numbers and expression:  # Only yield if we successfully generated a problem
            # Create prompt similar to test_inference
            numbers_str = ", ".join(map(str, numbers))
            prompt_content = f"Using the numbers {numbers_str} exactly once in mathematical notation using addition, subtraction, multiplication, division, and/or parentheses, create an expression that equals {target}. Keep your reasoning in the <think> block brief. Answer exactly in plain mathematical notation (DO NOT USE LATEX), WITH NO ADDITIONAL TEXT. For example, if the provided numbers are 8, 3, 2, 3, a valid answer would be: (3 / 3 + 2) * 8. Or, if the numbers were 8, 2, 9, 9, a valid answer would be 9 + 9 - 2 + 8. ANSWER AS SOON AS A CORRECT EXPRESSION IS FOUND. Do not include = {target} in your answer."
            
            # messages = [{"role": "user", "content": prompt_content}]
            
            # # Apply chat template with thinking mode enabled
            # prompt = tokenizer.apply_chat_template(
            #     messages,
            #     tokenize=False,
            #     add_generation_prompt=True,
            #     enable_thinking=True
            # )
            
            yield {
                "prompt": prompt_content,
                "target": target,
                "numbers": numbers
            }

def extract_numbers_from_expression(expression):
    """
    Extracts all numbers from a mathematical expression.
    
    Args:
        expression (str): Mathematical expression
        
    Returns:
        list: List of numbers found in the expression
    """
    # Remove all non-digit, non-decimal point, and non-space characters
    # Keep only numbers and spaces
    import re
    
    # Find all numbers (integers and decimals) in the expression
    numbers = re.findall(r'\d+(?:\.\d+)?', expression)
    
    # Convert to integers (assuming we're working with integers based on the problem setup)
    return [int(float(num)) for num in numbers]

def extract_numbers_from_prompt(prompt):
    """
    Extracts the list of numbers from the prompt.
    
    Args:
        prompt (str): The prompt containing the numbers
        
    Returns:
        list: List of numbers specified in the prompt
    """
    # Look for pattern "Using the numbers X, Y, Z, ..." in the prompt
    match = re.search(r'Using the numbers ([\d,\s]+)', prompt)
    
    if match:
        numbers_str = match.group(1)
        # Split by comma and convert to integers
        numbers = [int(num.strip()) for num in numbers_str.split(',')]
        return numbers
    
    return []

def numbers_match(prompt_numbers, expression_numbers):
    """
    Checks if the numbers used in the expression exactly match those specified in the prompt.
    
    Args:
        prompt_numbers (list): Numbers specified in the prompt
        expression_numbers (list): Numbers extracted from the expression
        
    Returns:
        bool: True if numbers match exactly (same numbers, same frequency)
    """
    # Sort both lists and compare
    return sorted(prompt_numbers) == sorted(expression_numbers)

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

        reward = 0.0  # Default to 0
        
        if target_match:
            target = float(target_match.group(1))
            
            # Extract numbers from prompt and expression
            prompt_numbers = extract_numbers_from_prompt(prompt)
            expression_numbers = extract_numbers_from_expression(content)
            
            # Check both mathematical correctness and number usage
            is_correct_answer = is_correct(content, target)
            numbers_are_correct = numbers_match(prompt_numbers, expression_numbers)
            
            # Only give full reward if both conditions are met
            if is_correct_answer and numbers_are_correct:
                reward = 1.0
            elif is_correct_answer and not numbers_are_correct:
                reward = 0.0  # Wrong numbers used
            elif not is_correct_answer and numbers_are_correct:
                reward = 0.0  # Right numbers but wrong result
            else:
                reward = 0.0  # Both wrong
        
        # Penalize completions that are clearly unfinished (unclosed <think> tag)
        if '<think>' in completion and '</think>' not in completion:
            reward = 0.0
            
        print("\n-----")
        print(f"Prompt: {prompt}")
        print(f"Completion: {completion}")
        print(f"Parsed Content: {content}")
        print(f"Prompt Numbers: {extract_numbers_from_prompt(prompt) if target_match else 'N/A'}")
        print(f"Expression Numbers: {extract_numbers_from_expression(content) if target_match else 'N/A'}")
        print(f"Numbers Match: {numbers_match(extract_numbers_from_prompt(prompt), extract_numbers_from_expression(content)) if target_match else 'N/A'}")
        print(f"Reward: {reward}")
        print("-----")
            
        rewards.append(reward)
    
    return rewards

def test_reward():
    """
    Test the reward function with some sample completions and prompts
    """
    test_completions = [
        "(4 + 4) * 3",      # Should be incorrect: right math (24) but wrong numbers (uses 3, 3, 4, 4 but expression has only one 3)
        "3 * 4 + 3 * 4",    # Should be correct: right math (24) and right numbers (3, 3, 4, 4)
        "4 + 4 + 3",        # Should be incorrect: wrong math (11) but uses some right numbers
        "10 * 2 + 4",       # Should be incorrect: right math (24) but wrong numbers (missing one 4, has 10 instead of specified numbers)
        "10 + 4 + 4 + 5 + 1", # Should be incorrect: right math (24) but wrong numbers (uses 5 numbers instead of 4)
        "(8 * (4 - 2)) + 8", # Should be correct: right math (24) and right numbers (8, 4, 2, 8)
        "6 * 4",            # Should be incorrect: right math (24) but completely wrong numbers
    ]
    
    test_prompts = [
        "Using the numbers 3, 3, 4, 4 exactly once in mathematical notation using addition, subtraction, multiplication, division, and parentheses, create an expression that equals 24.",
        "Using the numbers 3, 3, 4, 4 exactly once in mathematical notation using addition, subtraction, multiplication, division, and parentheses, create an expression that equals 24.", 
        "Using the numbers 3, 3, 4, 4 exactly once in mathematical notation using addition, subtraction, multiplication, division, and parentheses, create an expression that equals 24.",
        "Using the numbers 10, 2, 4, 4 exactly once in mathematical notation using addition, subtraction, multiplication, division, and parentheses, create an expression that equals 24.",
        "Using the numbers 10, 4, 4, 5 exactly once in mathematical notation using addition, subtraction, multiplication, division, and parentheses, create an expression that equals 24.",
        "Using the numbers 8, 4, 2, 8 exactly once in mathematical notation using addition, subtraction, multiplication, division, and parentheses, create an expression that equals 24.",
        "Using the numbers 1, 2, 3, 4 exactly once in mathematical notation using addition, subtraction, multiplication, division, and parentheses, create an expression that equals 24.",
    ]
    
    rewards = math_reward_func(test_completions, test_prompts)
    
    print("\nTesting reward function:")
    for i, (completion, prompt, reward) in enumerate(zip(test_completions, test_prompts, rewards)):
        print(f"\nTest {i+1}:")
        print(f"  Completion: {completion}")
        print(f"  Reward: {reward}")


if __name__ == "__main__":
    test_reward()

