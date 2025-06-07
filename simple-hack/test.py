from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import GRPOTrainer
import random

model_name = "Qwen/Qwen3-1.7B"

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
        print(f"Error evaluating expression '{content}': {e}")
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

    raw = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip("\n")
    # print("raw:", raw)

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("thinking content:", thinking_content)
    print("content:", content)

    # Check if the generated expression is correct
    result_is_correct = is_correct(content, 24)
    print(f"Is the answer correct? {result_is_correct}")


def reward_func(completions, **kwargs):
    # Dummy reward function that rewards completions with more unique letters.
    return [float(len(set(completion))) for completion in completions]

def train():
    # Define a dataset that contains both math and coding problems
    dataset = Dataset.from_list(
        [
            {"prompt": "What is 2+2?", "task": "math"},
            {"prompt": "Write a function that returns the sum of two numbers.", "task": "code"},
            {"prompt": "What is 3*4?", "task": "math"},
            {"prompt": "Write a function that returns the product of two numbers.", "task": "code"},
        ]
    )

    # Use both task-specific reward functions
    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=[reward_func],
        train_dataset=dataset,
    )

    trainer.train()