import pdb
import json
from datasets import Dataset

problems = json.load(open("/root/arc/ARC-AGI/GRPO-hack/easy_dataset/easiest_arc_problems150.json", "r"))

def grid_to_string(grid):
    # Convert each row to a string representation with brackets
    rows_str = [str(row) for row in grid]
    # Join rows with newline characters
    grid_str = "\n".join(rows_str)
    return grid_str

def generate_prompt(challenge):
    prompt = "You are a genius at solving IQ tests.\n\nBelow is a list of input and output pairs with a pattern. Your goal is to identify the pattern or transformation in the training examples that maps the input to the output, then apply that pattern to the test input to give a final output. This is not a maths puzzle, the numbers represent colors. This is like a visual IQ test.\n\nRespond in the format of the training output examples\n\n--Training Examples--"

    for i in range(len(challenge["examples"]) - 1):
        prompt += f"\n--Example {i}-- \n\n INPUT: \n\n"
        prompt += grid_to_string(challenge["examples"][i][0])
        prompt += "\n\n\nOUTPUT: \n\n"
        prompt += grid_to_string(challenge["examples"][i][1])
        prompt += "\n\n"

    prompt += "\n\n--End of Training Examples--\n\n--Test Input--\n"
    prompt += grid_to_string(challenge["examples"][-1][0])
    prompt += "\n\n--End of Test Input--\n\nProvide answer in <answer> tag precisely so that I can parse and validate."

    return prompt, challenge["examples"][-1][1]


# tokenize the prompts and run percentile statistics
prompts = []
answers = []
for challenge in problems:
    prompt, answer = generate_prompt(challenge)
    prompts.append([
        {"role": "user", "content": prompt},
    ])
    answers.append(answer)
    
dataset = Dataset.from_dict({
    "prompt": prompts,
    "answer": answers
})
