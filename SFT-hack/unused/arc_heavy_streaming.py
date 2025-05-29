import json
from datasets import Dataset, load_dataset
import gc
from typing import Iterator, Dict, Any

def grid_to_string(grid):
    # Convert each row to a string representation with brackets, converting floats to ints
    rows_str = [str([int(val) for val in row]) for row in grid]
    # Join rows with newline characters
    grid_str = "\n".join(rows_str)
    return grid_str

def generate_prompt(challenge):
    user_prompt = "You are a genius at solving IQ tests.\n\nBelow is a list of input and output pairs with a pattern. Your goal is to identify the pattern or transformation in the training examples that maps the input to the output, then apply that pattern to the test input to give a final output. This is not a maths puzzle, the numbers represent colors. This is like a visual IQ test.\n\nRespond in the format of the training output examples\n\n--Training Examples--"

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

def process_dataset_in_chunks(chunk_size: int = 1000, max_examples: int = 100000):
    """Process dataset in chunks to manage memory usage."""
    
    # Load dataset with streaming
    dataset = load_dataset(
        "barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems",
        streaming=True
    )
    
    all_messages = []
    all_examples = []
    current_chunk = []
    total_processed = 0
    
    print("Starting streaming processing...")
    
    for challenge in dataset["train"]:
        if total_processed >= max_examples:
            break
            
        try:
            user_prompt, assistant_response = generate_prompt(challenge)
            
            current_chunk.append({
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response}
                ],
                "examples": challenge["examples"]
            })
            
            total_processed += 1
            
            if total_processed % 1000 == 0:
                print(f"Processed {total_processed} examples")
            
            # Process chunk when it reaches the specified size
            if len(current_chunk) >= chunk_size:
                # Add chunk data to main collections
                for item in current_chunk:
                    all_messages.append(item["messages"])
                    all_examples.append(item["examples"])
                
                print(f"Processed chunk with {len(current_chunk)} examples (Total: {total_processed})")
                
                # Clear current chunk to free memory
                current_chunk = []
                gc.collect()
                
        except Exception as e:
            print(f"Error processing example {total_processed}: {e}")
            continue
    
    # Process remaining items if any
    if current_chunk:
        for item in current_chunk:
            all_messages.append(item["messages"])
            all_examples.append(item["examples"])
        print(f"Processed final chunk with {len(current_chunk)} examples")
    
    return all_messages, all_examples, total_processed

def upload_dataset(all_messages, all_examples, total_processed, dataset_name="arc-agi-prompts"):
    """Upload processed data to HuggingFace Hub."""
    
    print(f"Creating HuggingFace dataset with {total_processed} examples...")
    
    # Create the dataset
    dataset_dict = {
        "messages": all_messages,
        "examples": all_examples
    }
    
    hf_dataset = Dataset.from_dict(dataset_dict)
    
    # Upload to HuggingFace Hub
    print(f"Uploading dataset to HuggingFace Hub as '{dataset_name}'...")
    hf_dataset.push_to_hub(dataset_name, private=False)
    
    print(f"Successfully uploaded dataset with {total_processed} examples!")

def main():
    """Main function with chunked processing."""
    
    try:
        # Process in chunks
        all_messages, all_examples, total_processed = process_dataset_in_chunks(
            chunk_size=1000,  # Process 1000 examples at a time
            max_examples=150000
        )
        
        # Upload to HuggingFace Hub
        upload_dataset(all_messages, all_examples, total_processed)
        
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main() 