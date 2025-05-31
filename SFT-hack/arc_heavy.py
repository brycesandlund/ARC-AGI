import pdb
import json
from datasets import Dataset, load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load the HuggingFace dataset
dataset = load_dataset("barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems")

print(f"Dataset loaded. Train split has {len(dataset['train'])} examples")

def grid_to_string(grid):
    # Convert each row to a string representation with brackets, converting floats to ints
    rows_str = [str([int(val) for val in row]) for row in grid]
    # Join rows with newline characters
    grid_str = "\n".join(rows_str)
    return grid_str

def generate_prompt(challenge):
    user_prompt = "You are a genius at solving IQ tests.\n\nBelow is a list of input and output pairs with a pattern. Your goal is to identify the pattern or transformation in the training examples that maps the input to the output, then apply that pattern to the test input to give a final output. This is not a maths puzzle, the numbers represent colors. This is like a visual IQ test.\n\nRespond in the format of the training output examples\n\n--Training Examples--"

    # print("="*100)
    # print(challenge)
    # print("="*100)
    for i in range(len(challenge["examples"]) - 1):
        user_prompt += f"\n--Example {i}-- \n\n INPUT: \n\n"
        user_prompt += grid_to_string(challenge["examples"][i][0])
        user_prompt += "\n\n\nOUTPUT: \n\n"
        user_prompt += grid_to_string(challenge["examples"][i][1])
        user_prompt += "\n\n"

    user_prompt += "\n\n--End of Training Examples--\n\n<test_input>"
    user_prompt += grid_to_string(challenge["examples"][-1][0])
    user_prompt += "</test_input>"
    
    # Create the assistant response with the answer
    assistant_response = "<answer>"
    assistant_response += grid_to_string(challenge["examples"][-1][1])
    assistant_response += "</answer>"

    return user_prompt, assistant_response

def process_challenge(challenge):
    """Process a single challenge and return the prompt if it meets criteria"""
    try:
        user_prompt, assistant_response = generate_prompt(challenge)
        
        # Skip if user_prompt is larger than 30000 characters
        if len(user_prompt) > 15000:
            return None
        
        return {
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response}
            ],
            "examples": challenge["examples"]  # Keep the original examples
        }
    except Exception as e:
        print(f"Error processing challenge: {e}")
        return None

# Generate prompts from the first 300k examples of the HuggingFace dataset using multithreading
prompts = []
processed_count = 0
skipped_count = 0
lock = threading.Lock()

def update_progress(processed, skipped):
    """Thread-safe progress update"""
    with lock:
        if processed % 1000 == 0:
            print(f"Processed {processed} examples, kept {len(prompts)}, skipped {skipped} (too long)")

# Process in batches to avoid memory issues
batch_size = 1000
max_examples = min(300000, len(dataset["train"]))
max_workers = min(16, batch_size)  # Reduce max workers to avoid overwhelming the system

print(f"Starting processing of {max_examples} examples in batches of {batch_size}")

for batch_start in range(0, max_examples, batch_size):
    batch_end = min(batch_start + batch_size, max_examples)
    batch_challenges = [dataset["train"][i] for i in range(batch_start, batch_end)]
    
    print(f"Processing batch {batch_start//batch_size + 1}/{(max_examples-1)//batch_size + 1} (examples {batch_start}-{batch_end-1})")
    
    # Use ThreadPoolExecutor for parallel processing of this batch
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks for this batch
        future_to_challenge = {
            executor.submit(process_challenge, challenge): challenge 
            for challenge in batch_challenges
        }
        
        for future in as_completed(future_to_challenge):
            result = future.result()
            processed_count += 1
            
            if result is not None:
                prompts.append(result)
            else:
                skipped_count += 1
            
            update_progress(processed_count, skipped_count)

print(f"Final stats: Processed {processed_count} examples, kept {len(prompts)}, skipped {skipped_count}")

# # Save the prompts to a JSON file
# with open("arc_prompts.json", "w") as f:
#     json.dump(prompts, f, indent=2)

# print(f"Saved {len(prompts)} prompts to arc_prompts.json")

# Create a HuggingFace dataset from the prompts
dataset_dict = {
    "messages": [prompt["messages"] for prompt in prompts],
    "examples": [prompt["examples"] for prompt in prompts]
}

# Create the dataset
hf_dataset = Dataset.from_dict(dataset_dict)

# Push to HuggingFace Hub
# You'll need to be logged in with `huggingface-cli login` or set HF_TOKEN environment variable
dataset_name = "arc-agi-prompts"  # Change this to your desired dataset name
hf_dataset.push_to_hub(dataset_name, private=False)  # Set private=True if you want a private dataset

print(f"Published dataset to HuggingFace Hub as '{dataset_name}'")
