import os
import json
import random
import argparse
import re
from datasets import load_from_disk
import anthropic
from pprint import pprint

# Load API key from environment variable
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not CLAUDE_API_KEY:
    raise ValueError("Please set the ANTHROPIC_API_KEY environment variable")

# Initialize Claude client
client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

def get_messages_from_example(example):
    """Get messages from the example in the correct format for Claude API"""
    messages = []
    
    # Format the prompt with ARC examples
    prompt = format_arc_prompt(example)
    
    # Add as user message
    messages.append({
        "role": "user", 
        "content": prompt
    })
    
    return messages

def format_grid(grid_data):
    """Format a grid (list of lists) into a string with space-separated numbers and newlines"""
    if not grid_data:
        return ""
    
    # Check if grid_data is already a string
    if isinstance(grid_data, str):
        return grid_data
    
    # Convert grid data to string representation
    rows = []
    for row in grid_data:
        if isinstance(row, list):
            rows.append(" ".join(str(cell) for cell in row))
        else:
            # If it's not a list, just convert to string
            rows.append(str(row))
    
    return "\n".join(rows)

def format_arc_prompt(example):
    """Format an ARC example into the prompt format similar to one_arc_dataset.py"""
    
    # Get data from the example
    train_inputs = example.get("raw_train_inputs", [])
    train_outputs = example.get("raw_train_outputs", [])
    test_input = example.get("raw_test_inputs", [""])[0]  # Assuming there's at least one test input
    
    # Check if these are strings or need formatting
    formatted_train_inputs = []
    formatted_train_outputs = []
    
    # Format training inputs
    for train_input in train_inputs:
        formatted_train_inputs.append(format_grid(train_input))
    
    # Format training outputs
    for train_output in train_outputs:
        formatted_train_outputs.append(format_grid(train_output))
    
    # Format test input
    formatted_test_input = format_grid(test_input)
    
    # Build the prompt manually
    prompt = "Solve the following abstract reasoning challenge.  Here's the challenge:\n\n"
    prompt += "=== ALL TRAINING EXAMPLES ===\n\n"
    
    # Add all training examples
    for i in range(len(formatted_train_inputs)):
        prompt += f"Example {i+1}:\n"
        prompt += f"Input:\n{formatted_train_inputs[i]}\n\n"
        prompt += f"Output:\n{formatted_train_outputs[i]}\n\n"
        prompt += "=" * 40 + "\n\n"
    
    # Add test example
    prompt += "Test Example:\n"
    prompt += f"Input:\n{formatted_test_input}"
    
    return prompt

def call_claude(messages, temperature=0.0, max_tokens=4000):
    """Call Claude API with the given messages"""
    try:
        print(messages)
        # Pretty print messages with pprint
        pprint(messages)
        
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",  # Using Claude 3.7 Sonnet
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
            system="You are an expert at solving abstract reasoning puzzles. When solving ARC reasoning challenges, carefully look at all examples to identify patterns, then solve the test example step by step. First provide detailed reasoning in <reasoning> tags and then provide just the output grid in <answer> tags."
        )
        return response.content[0].text
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        return None

def extract_answer(response):
    """Extract the answer grid from Claude's response (inside <answer> tags)"""
    if not response:
        return None
    
    # Find content between <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
        return answer_text
    
    return None

def format_for_comparison(grid_text):
    """Format a grid text for comparison by normalizing whitespace"""
    if not grid_text:
        return None
    
    # Normalize whitespace, strip empty lines
    lines = [line.strip() for line in grid_text.strip().split('\n')]
    lines = [line for line in lines if line]  # Remove empty lines
    
    # Normalize spacing between numbers
    normalized_lines = []
    for line in lines:
        # Replace multiple spaces with single spaces
        normalized_line = re.sub(r'\s+', ' ', line)
        normalized_lines.append(normalized_line)
    
    return '\n'.join(normalized_lines)

def compare_grids(predicted, expected):
    """Compare predicted grid with expected grid after normalizing both"""
    if not predicted or not expected:
        return False
    
    # Format both grids for comparison
    formatted_predicted = format_for_comparison(predicted)
    formatted_expected = format_for_comparison(expected)
    
    # Print the comparison
    print("\nExpected Solution:")
    print(formatted_expected)
    print("\nClaude's Solution:")
    print(formatted_predicted)
    
    # Check if they match
    if formatted_predicted == formatted_expected:
        print("\n✓ CORRECT: Claude's solution matches expected output")
        return True
    else:
        print("\n✗ INCORRECT: Claude's solution does not match expected output")
        return False

def save_results(example_id, messages, response, output_dir="results", extra_data=None):
    """Save the messages and response to a file with optional extra data"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get the ARC challenge prompt
    arc_prompt = messages[0]["content"] if messages else None
    
    # Prepare the data to save
    result_data = {
        "id": example_id,
        "prompt": arc_prompt,
        "response": response
    }
    
    # Add any extra data if provided
    if extra_data:
        result_data.update(extra_data)
    
    with open(f"{output_dir}/{example_id}.json", "w") as f:
        json.dump(result_data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Run Claude on ARC dataset')
    parser.add_argument('--dataset_path', type=str, default='arc_small_input_output_hf_split/SFT', 
                        help='Path to the dataset')
    parser.add_argument('--num_examples', type=int, default=1, 
                        help='Number of examples to run (0 for all)')
    parser.add_argument('--temperature', type=float, default=0.0, 
                        help='Temperature for Claude API')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='Directory to save results')
    parser.add_argument('--random_seed', type=int, default=42, 
                        help='Random seed for example selection')
    parser.add_argument('--show_example', action='store_true',
                        help='Show formatted messages for the first example without calling API')
    parser.add_argument('--dry_run', action='store_true',
                        help='Only print example info without making API calls')
    
    args = parser.parse_args()
    
    # Load the dataset
    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    print(f"Dataset loaded with {len(dataset)} examples")
    
    # Print dataset structure for the first example
    if len(dataset) > 0:
        print("\nDataset structure (first example):")
        first_example = dataset[0]
        for key in first_example:
            if isinstance(first_example[key], list):
                print(f"- {key}: list with {len(first_example[key])} items")
            else:
                print(f"- {key}: {type(first_example[key])}")
    
    # Set random seed for reproducibility
    random.seed(args.random_seed)
    
    # Select examples
    if args.num_examples == 0 or args.num_examples >= len(dataset):
        examples_to_run = list(range(len(dataset)))
    else:
        examples_to_run = random.sample(range(len(dataset)), args.num_examples)
    
    # Optionally show the first example without running API call
    if (args.show_example or args.dry_run) and examples_to_run:
        example = dataset[examples_to_run[0]]
        example_id = example.get("id", f"example_{examples_to_run[0]}")
        messages = get_messages_from_example(example)
        
        print(f"\nExample {example_id} messages:")
        print("=" * 80)
        for msg in messages:
            print(f"Role: {msg['role']}")
            print(f"Content preview: {msg['content'][:500]}...")
            print("-" * 40)
        print("=" * 80)
        
        if args.dry_run or not input("\nContinue with API calls? (y/n): ").lower().startswith('y'):
            return
    
    # Process each example
    for i, idx in enumerate(examples_to_run):
        example = dataset[idx]
        example_id = example.get("id", f"example_{idx}")
        
        print(f"Processing example {example_id} ({i+1}/{len(examples_to_run)})")
        
        # Get messages from the example
        messages = get_messages_from_example(example)
        
        # Call Claude
        response = call_claude(messages, temperature=args.temperature)
        
        if response:
            # Extract answer from Claude's response
            claude_answer = extract_answer(response)
            
            # Get expected solution from dataset
            expected_solution = None
            if "raw_test_outputs" in example and example["raw_test_outputs"]:
                expected_solution = format_grid(example["raw_test_outputs"][0])
            
            # Compare Claude's answer with expected solution
            if expected_solution and claude_answer:
                is_correct = compare_grids(claude_answer, expected_solution)
                # Include the result in the saved data
                save_results(example_id, messages, response, args.output_dir, 
                             {"is_correct": is_correct, "claude_answer": claude_answer})
                print(f"Results saved to {args.output_dir}/{example_id}.json")
            else:
                save_results(example_id, messages, response, args.output_dir)
                print(f"Results saved to {args.output_dir}/{example_id}.json (No expected solution available)")
            
            # Print a preview of the response
            print("\nResponse preview:")
            print("-" * 40)
            preview = response[:200] + "..." if len(response) > 200 else response
            print(preview)
            print("-" * 40)
        else:
            print(f"Failed to get response for example {example_id}")
            
        break

if __name__ == "__main__":
    main()
