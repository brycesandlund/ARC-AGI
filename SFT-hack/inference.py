import torch
import json
import numpy as np
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TextStreamer
import re
from tqdm import tqdm
import os

# Configuration - OPTIMIZED FOR SPEED
BASE_MODEL_NAME = "bcsandlund/arc-model-unsloth-slow-lr-three-epochs"
DATASET_NAME = "bcsandlund/arc-agi-prompts-train-test-split"
MAX_NEW_TOKENS = 10000  # Increased to allow complete grid generation - grids can be large
TEMPERATURE = 0.7  # Enable sampling with moderate temperature
TOP_P = 0.9  # Enable nucleus sampling

def format_chatml_prompt(messages):
    """
    Format messages into ChatML format manually
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
    
    Returns:
        Formatted string in ChatML format
    """
    formatted_text = ""
    
    for message in messages:
        role = message.get('role', 'user')
        content = message.get('content', '')
        
        if role == 'user':
            formatted_text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == 'assistant':
            formatted_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        elif role == 'system':
            formatted_text += f"<|im_start|>system\n{content}<|im_end|>\n"
    
    return formatted_text

def extract_grid_from_answer(answer_text):
    """
    Extract grid from answer text that's in <answer>...</answer> tags
    
    Args:
        answer_text: Text containing the answer in tags
    
    Returns:
        List of lists representing the grid, or None if parsing failed
    """
    # Extract content between <answer> and </answer> tags
    match = re.search(r'<answer>(.*?)</answer>', answer_text, re.DOTALL)
    if not match:
        return None
    
    grid_text = match.group(1).strip()
    
    try:
        # Parse each line as a list
        lines = grid_text.strip().split('\n')
        grid = []
        for line in lines:
            line = line.strip()
            if line and line.startswith('[') and line.endswith(']'):
                # Convert string representation of list to actual list
                row = eval(line)  # Using eval for simplicity, could use ast.literal_eval for safety
                grid.append(row)
        return grid if grid else None
    except:
        return None

def grid_to_string(grid):
    """Convert grid to string representation matching the dataset format"""
    if not grid:
        return ""
    rows_str = [str(row) for row in grid]
    return "\n".join(rows_str)

def grids_equal(grid1, grid2):
    """Check if two grids are equal"""
    if grid1 is None or grid2 is None:
        return False
    
    if len(grid1) != len(grid2):
        return False
    
    for i in range(len(grid1)):
        if len(grid1[i]) != len(grid2[i]):
            return False
        for j in range(len(grid1[i])):
            if grid1[i][j] != grid2[i][j]:
                return False
    
    return True

def run_inference():
    """Main inference function"""
    print("Loading model and tokenizer...")
    
    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model and tokenizer with Unsloth (same as training)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=10000,  # Reduced for faster inference
        dtype=None,  # Auto-detect
        load_in_4bit=True,  # Use 4-bit quantization for memory efficiency
    )
    
    # Enable fast inference mode
    FastLanguageModel.for_inference(model)
    
    # Verify model is on GPU and check memory usage
    print(f"Model device: {next(model.parameters()).device}")
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    
    # Note: Model is already on correct device when loaded with quantization
    # No need to move it manually
    
    print("Loading test dataset...")
    dataset = load_dataset(DATASET_NAME)
    test_data = dataset['train']
    
    print(f"Test dataset size: {len(test_data)} examples")
    
    # Prepare for evaluation
    results = []
    correct_count = 0
    total_count = 0
    
    # Create output directory
    os.makedirs('./inference_results', exist_ok=True)
    
    print("Starting inference...")
    
    for i, example in enumerate(tqdm(test_data, desc="Running inference")):
        try:
            # Get the user message (question)
            messages = example['messages']
            user_message = None
            ground_truth_answer = None
            
            for msg in messages:
                if msg['role'] == 'user':
                    user_message = msg['content']
                elif msg['role'] == 'assistant':
                    ground_truth_answer = msg['content']
            
            if not user_message or not ground_truth_answer:
                print(f"Skipping example {i}: missing user message or ground truth")
                continue
            
            # Format the prompt for inference (only user message)
            inference_messages = [{"role": "user", "content": user_message}]
            formatted_prompt = format_chatml_prompt(inference_messages)
            
            # Add the assistant start token to prompt for generation
            formatted_prompt += "<|im_start|>assistant\n"
            
            # Tokenize with proper attention mask
            inputs = tokenizer(
                formatted_prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=10000  # Reduced for faster processing
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # # Check if prompt is too long
            # if inputs.shape[1] > 1500:  # Leave room for generation
            #     print(f"Skipping example {i}: prompt too long ({inputs.shape[1]} tokens)")
            #     continue
            
            # Generate response with optimized parameters
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if start_time:
                start_time.record()
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,  # Use sampling with temperature
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.0,  # Disable repetition penalty for speed
                    use_cache=True,  # Enable KV cache for faster generation
                )
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                generation_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
                if (i + 1) % 10 == 0:
                    print(f"Generation time: {generation_time:.2f}s")
            
            # Decode the generated response
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

            print(f"Generated text: {generated_text}")
            print("\n\n")

            
            # Extract just the new assistant's response (only the newly generated part)
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]  # Get only the newly generated tokens
            
            # Debug: Check if generation hit the token limit
            generated_length = len(generated_tokens)
            hit_token_limit = generated_length >= MAX_NEW_TOKENS
            
            assistant_response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Remove any trailing special tokens that might have been generated
            assistant_response = assistant_response.replace("<|im_end|>", "").strip()
            
            # Check if response seems incomplete (missing closing answer tag)
            has_opening_tag = "<answer>" in assistant_response
            has_closing_tag = "</answer>" in assistant_response
            incomplete_answer = has_opening_tag and not has_closing_tag
            
            # Debug: Print raw generation info for first few examples
            if i < 3 or hit_token_limit or incomplete_answer:
                print(f"DEBUG - Example {i}:")
                print(f"  Generated tokens: {generated_length}/{MAX_NEW_TOKENS}")
                print(f"  Hit token limit: {hit_token_limit}")
                print(f"  Has opening <answer> tag: {has_opening_tag}")
                print(f"  Has closing </answer> tag: {has_closing_tag}")
                print(f"  Incomplete answer: {incomplete_answer}")
                print(f"  Raw assistant response: {repr(assistant_response)}")
                if hit_token_limit or incomplete_answer:
                    print(f"  WARNING: Generation may be truncated!")
                print()
            
            # Extract grids from both predicted and ground truth answers
            predicted_grid = extract_grid_from_answer(assistant_response)
            ground_truth_grid = extract_grid_from_answer(ground_truth_answer)

            print(f"Predicted grid: {predicted_grid}")
            print("\n\n")
            print(f"Ground truth grid: {ground_truth_grid}")
            print("\n\n")
            
            # Check if prediction is correct
            is_correct = grids_equal(predicted_grid, ground_truth_grid)
            
            if is_correct:
                correct_count += 1
            total_count += 1
            
            # Store result
            result = {
                'example_id': i,
                'user_prompt': user_message,
                'ground_truth': ground_truth_answer,
                'predicted_response': assistant_response,
                'predicted_grid': predicted_grid,
                'ground_truth_grid': ground_truth_grid,
                'is_correct': is_correct,
                'prompt_length': inputs['input_ids'].shape[1]
            }
            results.append(result)
            
            # Print progress every 10 examples
            if (i) % 1 == 0:
                accuracy = correct_count / total_count if total_count > 0 else 0
                print(f"Processed {total_count} examples, Accuracy: {accuracy:.3f}")
                print(f"\n" + "-"*50)
                print(f"EXAMPLE {i + 1} DETAILS:")
                print(f"Prompt: {user_message}")
                print(f"\nGround Truth Answer: {ground_truth_answer}")
                print(f"\nModel Answer: {assistant_response}")
                print(f"Generated {generated_length} tokens (limit: {MAX_NEW_TOKENS})")
                if hit_token_limit:
                    print("⚠️  WARNING: Response may be truncated due to token limit!")
                if incomplete_answer:
                    print("⚠️  WARNING: Response appears incomplete (missing closing </answer> tag)!")
                print(f"Correct: {is_correct}")
                print("-"*50 + "\n")
            
            # Save results periodically
            if (i + 1) % 50 == 0:
                with open('./inference_results/partial_results.json', 'w') as f:
                    json.dump(results, f, indent=2)
        
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            continue
    
    # Calculate final accuracy
    final_accuracy = correct_count / total_count if total_count > 0 else 0
    
    print(f"\n" + "="*60)
    print("INFERENCE RESULTS")
    print("="*60)
    print(f"Total examples processed: {total_count}")
    print(f"Correct predictions: {correct_count}")
    print(f"Accuracy: {final_accuracy:.3f} ({final_accuracy*100:.1f}%)")
    
    # Save final results
    final_results = {
        'summary': {
            'total_examples': total_count,
            'correct_predictions': correct_count,
            'accuracy': final_accuracy,
            'model_name': BASE_MODEL_NAME,
            'dataset_name': DATASET_NAME
        },
        'detailed_results': results
    }
    
    with open('./inference_results/final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"Results saved to ./inference_results/final_results.json")
    
    # Print some example results
    print(f"\n" + "="*60)
    print("SAMPLE RESULTS")
    print("="*60)
    
    correct_examples = [r for r in results if r['is_correct']]
    incorrect_examples = [r for r in results if not r['is_correct']]
    
    if correct_examples:
        print(f"\nExample of CORRECT prediction:")
        example = correct_examples[0]
        print(f"Ground truth grid:\n{grid_to_string(example['ground_truth_grid'])}")
        print(f"Predicted grid:\n{grid_to_string(example['predicted_grid'])}")
    
    if incorrect_examples:
        print(f"\nExample of INCORRECT prediction:")
        example = incorrect_examples[0]
        print(f"Ground truth grid:\n{grid_to_string(example['ground_truth_grid'])}")
        print(f"Predicted grid:\n{grid_to_string(example['predicted_grid'])}")
        print(f"Raw prediction:\n{example['predicted_response']}")

    return final_accuracy, results

if __name__ == "__main__":
    accuracy, results = run_inference() 