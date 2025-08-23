from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import random
import re
from typing import List, Dict, Any, Optional

from enums import ModelType

LOG_FREQUENCY = 1 # Print logs every 50 calls on average

# Set a seed for reproducibility
random.seed(42)

def parse_completion(completion: str, model_type: ModelType) -> tuple[str, str]:
    """
    Parses a completion string that may contain <think>...</think> or <reasoning>...</reasoning> blocks.
    Extracts thinking/reasoning content and the final answer.
    
    Args:
        completion (str): The model's completion string.
        model_type (ModelType): The type of model, which determines the tags to use.
        
    Returns:
        tuple[str, str]: A tuple of (thinking_content, final_content).
    """
    if model_type == ModelType.THINKING:
        start_tag = '<think>'
        end_tag = '</think>'
    else:
        start_tag = '<reasoning>'
        end_tag = '</reasoning>'
    
    end_tag_pos = completion.rfind(end_tag)
    
    if end_tag_pos != -1:
        content = completion[end_tag_pos + len(end_tag):].strip()
        
        # Extract thinking content
        think_part = completion[:end_tag_pos]
        start_tag_pos = think_part.rfind(start_tag)
        
        if start_tag_pos != -1:
            thinking_content = think_part[start_tag_pos + len(start_tag):].strip()
        else:
            # If no start tag is found, assume everything before the end tag is thinking.
            thinking_content = think_part.strip()
        
        return thinking_content, content
    
    # No end tag found
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

def generate_math_problems(tokenizer, dataset_size, model_type: ModelType):
    """
    Generator function that creates math problems using generate_problem.
    Yields dictionary with 'prompt' containing the problem description, formatted with thinking template.
    """
    targets = [16]  # Various target numbers
    
    for _ in range(dataset_size):
        target = random.choice(targets)
        num_count = random.choice([2, 3, 4])
        numbers, expression = generate_problem(target, num_count=num_count, num_range=10)
        
        if numbers and expression:  # Only yield if we successfully generated a problem
            # Create prompt similar to test_inference
            numbers_str = ", ".join(map(str, numbers))

            if model_type == ModelType.BASE:
                # # R1-ZERO PROMPT:
                # prompt_content = (
                #     "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
                #     "The assistant first thinks about the reasoning process in the <think></think> tags and then provides the user with the answer.\n"
                #     f"User: Using the numbers {numbers_str} exactly once in mathematical notation using addition, "
                #     f"subtraction, multiplication, division, and/or parentheses, create an expression that equals {target}."
                #     "Show your work in <think> </think> tags. Output your answer after closing the </think> tag WITH NO ADDITIONAL TEXT.For example, if the provided numbers are 8, 3, 2, 3, a valid response would be Assistant: <think> Let me solve this step by step. Hm, maybe I can use 3 / 3 + 2 to get 3. Then I can multiply that by 8 to get 24. </think>(3 / 3 + 2) * 8.\n"
                #     "Assistant: <think> Let me solve this step by step. "
                # )



                # prompt_content = (
                #     f"SYSTEM: Using the numbers 8, 3, 2, 3, exactly once in mathematical notation using addition, subtraction, multiplication, division, and/or parentheses, create an expression that equals {target}. Show your"
                #     "work in the <think> </think> tags. Answer exactly in plain mathematical notation, WITH NO ADDITIONAL TEXT. \n\n"
                #     "ASSISTANT: <think> Let me solve this step by step. \n\nHm, maybe I can use 3 / 3 + 2 to get 3. Then I can multiply that by 8 to get 24. </think>(3 / 3 + 2) * 8.\n\n"
                #     f"SYSTEM: Using the numbers 4, 1, 8, 8, exactly once in mathematical notation using addition, subtraction, multiplication, division, and/or parentheses, create an expression that equals {target}. Show your"
                #     "work in the <think> </think> tags. Answer exactly in plain mathematical notation, WITH NO ADDITIONAL TEXT. \n\n"
                #     "ASSISTANT: <think> Let me solve this step by step. \n\nOkay, let's see. I need to use the numbers 4, 1, 8, 8 exactly once each with basic operations to make 24. 8 * 4 = 32. Then 32 minus something. If I can get 8 from 1 and 8. Wait, 8 - (8/ something). Wait, 8 - (8/ something). But I already used one 8 in 8*4. So maybe 8*4 - (8 - 1) = 32 -7=25. Close, but no. How about 8*4 - (8/(1))? That's 32 -8=24. Wait! Let me check. 8 multiplied by 4 is 32. Then 8 divided by 1 is 8. So 32 -8=24. And we've used 8, 4, 8, and 1. Exactly once each. So that works! </think>8 * 4 - 8 / 1\n"
                #     f"SYSTEM: Using the numbers 3, 9, 6, 2, exactly once in mathematical notation using addition, subtraction, multiplication, division, and/or parentheses, create an expression that equals {target}. Show your"
                #     "work in the <think> </think> tags. Answer exactly in plain mathematical notation, WITH NO ADDITIONAL TEXT. \n\n"
                #     "ASSISTANT: <think> Let me solve this step by step. \n\n6 * 4 = 24. How to get 4 from 3, 9, 2? 9 - 3 - 2 = 4. Yes! So 6 * (9 - 3 - 2) = 6 * 4 = 24. Let me check: 9-3=6, 6-2=4. Then 6*4=24. And we've used 6, 9, 3, 2. All four numbers exactly once. Perfect! </think>6*(9-3-2)\n"
                #     f"SYSTEM: Using the numbers {numbers_str} exactly once in mathematical notation using addition, subtraction, multiplication, division, and/or parentheses, create an expression that equals {target}. Show your"
                #     "work in the <think> </think> tags. Answer exactly in plain mathematical notation, WITH NO ADDITIONAL TEXT. \n\n"
                #     "ASSISTANT: <think> Let me solve this step by step. \n\n"
                # )



                prompt_content = (
                    f"Q: Using the numbers 2, 8, 10 exactly once in mathematical notation using addition, subtraction, multiplication, division, and/or parentheses, create an expression that equals {target}. Show your "
                    "work in the <reasoning> </reasoning> tags. \n\n"
                    "A: <reasoning> Let me solve this step by step. Hm, 10+8=18. and 18-2=16. That's it!</think>10+8-2\n\n"
                    f"Q: Using the numbers 10, 6 exactly once in mathematical notation using addition, subtraction, multiplication, division, and/or parentheses, create an expression that equals {target}. Show your "
                    "work in the <reasoning> </reasoning> tags. \n\n"
                    "A: <reasoning> Let me solve this step by step. Hm, 10+6 = 16.</think>10+6\n\n"
                    f"Q: Using the numbers 4, 3, 4 exactly once in mathematical notation using addition, subtraction, multiplication, division, and/or parentheses, create an expression that equals {target}. Show your "
                    "work in the <reasoning> </reasoning> tags. \n\n"
                    "A: <reasoning> Let me solve this step by step. Hm, 4*3 = 12. And 12+4=16.</think>4*3+4\n\n"
                    f"Q: Using the numbers 9, 2, 5, 3 exactly once in mathematical notation using addition, subtraction, multiplication, division, and/or parentheses, create an expression that equals {target}. Show your "
                    "work in the <reasoning> </reasoning> tags. \n\n"
                    "A: <reasoning> Let me solve this step by step. Hm, 5*2 = 10. And 9-3=6.</think>5*2+9-3\n\n"
                    f"Q: Using the numbers {numbers_str} exactly once in mathematical notation using addition, subtraction, multiplication, division, and/or parentheses, create an expression that equals {target}. Show your "
                    "work in the <reasoning> </reasoning> tags. \n\n"
                    "A: "
                )

                # prompt_content = (
                #     f"1, 2, 3, 4, 5, 6, "
                # )

            elif model_type == ModelType.INSTRUCT:
                # INSTRUCT-TUNED PROMPT (no thinking):
                prompt_content = f"Using the numbers {numbers_str} exactly once in mathematical notation using addition, subtraction, multiplication, division, and/or parentheses, create an expression that equals {target}. Show your work in the <think></think> tags. Answer exactly in plain mathematical notation, WITH NO ADDITIONAL TEXT. For example, if the provided numbers are 2, 8, 10, a valid answer would be: <think> Let me solve this step by step. Hm, 10+8=18. and 18-2=16. That's it!</think>10+8-2\n\n Or, if the numbers were 10, 6 a valid answer would be  <think> Let me solve this step by step. Hm, 10+6 = 16.</think>10+6\n\n Do not include = {target} in your answer."

                # prompt_content = (
                #     f"Your task is to use the numbers {numbers_str} exactly once to create a mathematical expression that equals {target}.\n\n"
                #     "**Allowed operations:** addition, subtraction, multiplication, division, parentheses.\n\n"
                #     "**Output format:**\n"
                #     "1. START WITH `<think>` tags. Inside, show your step-by-step reasoning.\n"
                #     "2. After the closing `</think>` tag, provide ONLY the final mathematical expression.\n\n"
                #     "**Crucial Rules:**\n"
                #     "- DO NOT include any other text, explanation, or the `= 16` part in your final answer.\n"
                #     "- DO NOT START YOUR RESPONSE WITH `<tool_call>`, `</tool_call>`, OR ANY OTHER TAG BESIDES `<think>`.\n"
                #     "- DO NOT use any numbers not listed in the provided set.\n\n"
                #     "**Example 1:**\n"
                #     "Numbers: 2, 8, 10\n"
                #     "<think>Let me solve this step by step. Hm, 10+8=18. and 18-2=16. That's it!</think>10+8-2\n\n"
                #     "**Example 2:**\n"
                #     "Numbers: 10, 6\n"
                #     "<think>Let me solve this step by step. Hm, 10+6 = 16.</think>10+6\n\n"
                #     f"**Your numbers:** {numbers_str}"
                # )

            else: # Catches ModelType.THINKING
                # REASONING-TRAINED PROMPT:
                prompt_content = f"Using the numbers {numbers_str} exactly once in mathematical notation using addition, subtraction, multiplication, division, and/or parentheses, create an expression that equals {target}. Keep your reasoning in the <think> block brief. Answer exactly in plain mathematical notation (DO NOT USE LATEX), WITH NO ADDITIONAL TEXT. For example, if the provided numbers are 8, 3, 2, 3, a valid answer would be: (3 / 3 + 2) * 8. Or, if the numbers were 8, 2, 9, 9, a valid answer would be 9 + 9 - 2 + 8. ANSWER AS SOON AS A CORRECT EXPRESSION IS FOUND. Do not include = {target} in your answer."
            
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

def math_reward_func(completions, prompts, numbers_list, model_type: ModelType, **kwargs):
    """
    Reward function that evaluates mathematical correctness using is_correct.
    
    Args:
        completions: List of generated completions
        prompts: List of prompts (to extract target values)
        numbers_list: List of lists of numbers used in prompts
        **kwargs: Additional arguments
        
    Returns:
        List of reward scores (1.0 for correct, 0.0 for incorrect)
    """
    rewards = []
    
    for completion, prompt, prompt_numbers in zip(completions, prompts, numbers_list):
        # Extract target from prompt
        # Look for "equals X" pattern in the prompt
        target_match = re.search(r'equals (\d+)', prompt)
        
        # Clean the completion and check if it's correct
        _, content = parse_completion(completion, model_type=model_type)

        reward = 0.0  # Default to 0
        
        if target_match:
            target = float(target_match.group(1))
            
            # Extract numbers from expression
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
        
        # If the answer isn't perfect, give partial credit for using thinking tags
        if reward < 1.0:
            partial_reward = 0.0
            if model_type == ModelType.THINKING:
                start_tag, end_tag = "<think>", "</think>"
            else:
                start_tag, end_tag = "<reasoning>", "</reasoning>"

            if start_tag in completion:
                partial_reward += 0.1
            if end_tag in completion:
                partial_reward += 0.1
            # if f"16" in completion:
            #     partial_reward += 0.05
            reward = partial_reward
            
        if random.random() < LOG_FREQUENCY:
            print("\n-----")
            print(f"Prompt: {prompt}")
            print(f"Completion: {completion}")
            print(f"Parsed Content: {content}")
            print(f"Prompt Numbers: {prompt_numbers if target_match else 'N/A'}")
            print(f"Expression Numbers: {extract_numbers_from_expression(content) if target_match else 'N/A'}")
            print(f"Numbers Match: {numbers_match(prompt_numbers, extract_numbers_from_expression(content)) if target_match else 'N/A'}")
            print(f"Reward: {reward}")
            print("-----")
            
        rewards.append(reward)
    
    return rewards