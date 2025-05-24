import pdb
import json
from datasets import Dataset, load_dataset

# Load the HuggingFace dataset
dataset = load_dataset("barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems")

def grid_to_string(grid):
    # Convert each row to a string representation with brackets, converting floats to ints
    rows_str = [str([int(val) for val in row]) for row in grid]
    # Join rows with newline characters
    grid_str = "\n".join(rows_str)
    return grid_str

def generate_prompt(challenge):
    prompt = "You are a genius at solving IQ tests.\n\nBelow is a list of input and output pairs with a pattern. Your goal is to identify the pattern or transformation in the training examples that maps the input to the output, then apply that pattern to the test input to give a final output. This is not a maths puzzle, the numbers represent colors. This is like a visual IQ test.\n\nRespond in the format of the training output examples\n\n--Training Examples--"

    print("="*100)
    print(challenge)
    print("="*100)
    for i in range(len(challenge["examples"]) - 1):
        prompt += f"\n--Example {i}-- \n\n INPUT: \n\n"
        prompt += grid_to_string(challenge["examples"][i][0])
        prompt += "\n\n\nOUTPUT: \n\n"
        prompt += grid_to_string(challenge["examples"][i][1])
        prompt += "\n\n"

    prompt += "\n\n--End of Training Examples--\n\n<test_input>"
    prompt += grid_to_string(challenge["examples"][-1][0])
    prompt += "</test_input>"
    
    # Include the answer in the prompt for supervised training
    prompt += "<answer>"
    prompt += grid_to_string(challenge["examples"][-1][1])
    prompt += "</answer>"

    return prompt

# Generate prompts from the first 10 examples of the HuggingFace dataset
prompts = []

i = 0
for challenge in dataset["train"]:  # Only take first 10 examples
    # pdb.set_trace()
    prompt = generate_prompt(challenge)
    prompts.append({
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "examples": challenge["examples"]  # Keep the original examples
    })
    i += 1
    if i > 10:
        break

# Save the prompts to a JSON file
with open("arc_prompts.json", "w") as f:
    json.dump(prompts, f, indent=2)

print(f"Saved {len(prompts)} prompts to arc_prompts.json")

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
