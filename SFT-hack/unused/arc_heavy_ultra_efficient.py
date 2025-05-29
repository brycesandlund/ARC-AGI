import json
import tempfile
import os
from datasets import Dataset, load_dataset, DatasetDict
import gc
from huggingface_hub import HfApi
import time

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

def create_jsonl_file(chunk_size: int = 1000):
    """Create a JSONL file for efficient streaming to HuggingFace."""
    
    # Load dataset with streaming
    dataset = load_dataset(
        "barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems",
        streaming=True
    )
    
    jsonl_filename = "arc_prompts.jsonl"
    total_processed = 0
    
    print("Creating JSONL file for efficient upload...")
    
    with open(jsonl_filename, 'w') as f:
        for challenge in dataset["train"]:
            try:
                user_prompt, assistant_response = generate_prompt(challenge)
                
                # Create a single record
                record = {
                    "messages": [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": assistant_response}
                    ],
                    "examples": challenge["examples"]
                }
                
                # Write as JSONL (one JSON object per line)
                f.write(json.dumps(record) + '\n')
                
                total_processed += 1
                
                if total_processed % 100 == 0:
                    print(f"Processed {total_processed} examples")
                    # Flush to disk periodically
                    f.flush()
                
                # Force garbage collection periodically
                if total_processed % 1000 == 0:
                    gc.collect()
                    
            except Exception as e:
                print(f"Error processing example {total_processed}: {e}")
                continue
    
    print(f"Created JSONL file with {total_processed} examples")
    return jsonl_filename, total_processed

def upload_from_jsonl(jsonl_filename, dataset_name="arc-agi-prompts", chunk_size=1000):
    """Upload dataset from JSONL file in chunks."""
    
    print(f"Loading dataset from {jsonl_filename} and uploading to HuggingFace...")
    
    # Read JSONL and create dataset
    data = {"messages": [], "examples": []}
    
    with open(jsonl_filename, 'r') as f:
        for line_num, line in enumerate(f):
            if line.strip():  # Skip empty lines
                try:
                    record = json.loads(line)
                    data["messages"].append(record["messages"])
                    data["examples"].append(record["examples"])
                    
                    # Upload in chunks to avoid memory issues
                    if len(data["messages"]) >= chunk_size:
                        print(f"Uploading chunk of {len(data['messages'])} examples...")
                        
                        # Create mini dataset and upload
                        chunk_dataset = Dataset.from_dict(data)
                        
                        if line_num < chunk_size:  # First chunk - create new dataset
                            chunk_dataset.push_to_hub(dataset_name, private=False)
                        else:  # Subsequent chunks - append to existing
                            # Note: HuggingFace doesn't have direct append, so we'll create with suffix
                            chunk_name = f"{dataset_name}_chunk_{line_num // chunk_size}"
                            chunk_dataset.push_to_hub(chunk_name, private=False)
                        
                        # Clear data for next chunk
                        data = {"messages": [], "examples": []}
                        gc.collect()
                        
                        # Small delay to avoid rate limiting
                        time.sleep(1)
                        
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue
    
    # Upload remaining data if any
    if data["messages"]:
        print(f"Uploading final chunk of {len(data['messages'])} examples...")
        final_dataset = Dataset.from_dict(data)
        if len(data["messages"]) < 1000:  # Small final chunk
            final_dataset.push_to_hub(f"{dataset_name}_final", private=False)
        else:
            final_dataset.push_to_hub(dataset_name, private=False)
    
    # Clean up JSONL file
    os.remove(jsonl_filename)
    print(f"Successfully uploaded dataset and cleaned up temporary files!")

def main_ultra_efficient():
    """Ultra memory-efficient processing using JSONL intermediate format."""
    
    try:
        # Create JSONL file (streams data, minimal memory usage)
        jsonl_file, total_processed = create_jsonl_file(chunk_size=500)
        
        print(f"Now uploading {total_processed} examples to HuggingFace...")
        
        # Upload from JSONL in chunks
        upload_from_jsonl(jsonl_file, chunk_size=1000)
        
    except Exception as e:
        print(f"Error during processing: {e}")
        # Clean up if needed
        if 'jsonl_file' in locals() and os.path.exists(jsonl_file):
            os.remove(jsonl_file)

def main_simple():
    """Simplified version that creates a single dataset (requires more memory but simpler)."""
    
    print("Using simplified approach - loading in batches...")
    
    # Load dataset with streaming
    dataset = load_dataset(
        "barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems",
        streaming=True
    )
    
    batch_size = 2000  # Adjust based on your available memory
    all_messages = []
    all_examples = []
    total_processed = 0
    
    current_batch = []
    
    for challenge in dataset["train"]:
        try:
            user_prompt, assistant_response = generate_prompt(challenge)
            
            current_batch.append({
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response}
                ],
                "examples": challenge["examples"]
            })
            
            total_processed += 1
            
            if total_processed % 100 == 0:
                print(f"Processed {total_processed} examples")
            
            # Process batch when full
            if len(current_batch) >= batch_size:
                # Add to main collections
                for item in current_batch:
                    all_messages.append(item["messages"])
                    all_examples.append(item["examples"])
                
                # Clear batch and force garbage collection
                current_batch = []
                gc.collect()
                
                print(f"Accumulated {len(all_messages)} examples so far...")
                
        except Exception as e:
            print(f"Error processing example {total_processed}: {e}")
            continue
    
    # Process remaining items
    if current_batch:
        for item in current_batch:
            all_messages.append(item["messages"])
            all_examples.append(item["examples"])
    
    print(f"Creating final dataset with {len(all_messages)} examples...")
    
    # Create and upload final dataset
    dataset_dict = {
        "messages": all_messages,
        "examples": all_examples
    }
    
    hf_dataset = Dataset.from_dict(dataset_dict)
    
    # Clear memory before upload
    del all_messages, all_examples, dataset_dict
    gc.collect()
    
    # Upload
    dataset_name = "arc-agi-prompts"
    print(f"Uploading to HuggingFace Hub as '{dataset_name}'...")
    hf_dataset.push_to_hub(dataset_name, private=False)
    
    print(f"Successfully uploaded dataset with {total_processed} examples!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--ultra":
        main_ultra_efficient()
    else:
        main_simple() 