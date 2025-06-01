import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from collections import Counter
import statistics
from huggingface_hub import login, HfApi
import os
from sklearn.model_selection import train_test_split

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

def format_dataset_for_sft(example):
    """
    Format dataset examples for SFT training
    
    Args:
        example: Dataset example containing messages or other fields
    
    Returns:
        Dictionary with 'text' field containing formatted prompt
    """
    if 'messages' in example and example['messages']:
        # Use the custom ChatML formatting function
        formatted_text = format_chatml_prompt(example['messages'])
        return {"text": formatted_text}
    
    elif 'text' in example:
        # If already has text field, use it directly
        return {"text": example['text']}
    
    else:
        # Fallback: try to create a simple format from available fields
        content = ""
        for key, value in example.items():
            if isinstance(value, str):
                content += f"{key}: {value}\n"
        
        # Wrap in basic ChatML format
        formatted_text = f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\nProcessing...<|im_end|>\n"
        return {"text": formatted_text}

def analyze_token_lengths(dataset, tokenizer, sample_size=None):
    """
    Analyze token length statistics for the dataset
    
    Args:
        dataset: The formatted dataset
        tokenizer: The tokenizer to use
        sample_size: Optional sample size for analysis (None for full dataset)
    
    Returns:
        Dictionary containing various statistics
    """
    print("Tokenizing dataset...")
    
    # Sample dataset if specified
    if sample_size and len(dataset) > sample_size:
        dataset = dataset.select(range(sample_size))
        print(f"Using sample of {sample_size} examples")
    
    token_lengths = []
    
    for i, example in enumerate(dataset):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(dataset)} examples")
        
        text = example['text']
        tokens = tokenizer.encode(text, add_special_tokens=True)
        token_lengths.append(len(tokens))
    
    # Calculate statistics
    token_lengths = np.array(token_lengths)
    
    stats = {
        'count': len(token_lengths),
        'mean': np.mean(token_lengths),
        'median': np.median(token_lengths),
        'std': np.std(token_lengths),
        'min': np.min(token_lengths),
        'max': np.max(token_lengths),
        'percentiles': {
            '1st': np.percentile(token_lengths, 1),
            '5th': np.percentile(token_lengths, 5),
            '10th': np.percentile(token_lengths, 10),
            '25th': np.percentile(token_lengths, 25),
            '50th': np.percentile(token_lengths, 50),
            '75th': np.percentile(token_lengths, 75),
            '90th': np.percentile(token_lengths, 90),
            '95th': np.percentile(token_lengths, 95),
            '99th': np.percentile(token_lengths, 99),
        }
    }
    
    return stats, token_lengths

def print_detailed_stats(stats):
    """
    Print detailed statistics in a formatted way
    """
    print("\n" + "="*60)
    print("TOKEN LENGTH ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\nBasic Statistics:")
    print(f"  Total examples: {stats['count']:,}")
    print(f"  Mean length: {stats['mean']:.2f} tokens")
    print(f"  Median length: {stats['median']:.2f} tokens")
    print(f"  Standard deviation: {stats['std']:.2f} tokens")
    print(f"  Minimum length: {stats['min']} tokens")
    print(f"  Maximum length: {stats['max']} tokens")
    
    print(f"\nPercentile Analysis:")
    for percentile, value in stats['percentiles'].items():
        print(f"  {percentile} percentile: {value:.1f} tokens")
    
    # Additional insights
    print(f"\nInsights:")
    print(f"  Range: {stats['max'] - stats['min']} tokens")
    print(f"  Interquartile Range (IQR): {stats['percentiles']['75th'] - stats['percentiles']['25th']:.1f} tokens")
    print(f"  Coefficient of Variation: {(stats['std'] / stats['mean']) * 100:.1f}%")
    
    # Recommendations for sequence length
    print(f"\nSequence Length Recommendations:")
    print(f"  For 90% coverage: {int(stats['percentiles']['90th'])} tokens")
    print(f"  For 95% coverage: {int(stats['percentiles']['95th'])} tokens")
    print(f"  For 99% coverage: {int(stats['percentiles']['99th'])} tokens")

def analyze_by_message_role(dataset, tokenizer):
    """
    Analyze token lengths by message role (user vs assistant)
    """
    print("\nAnalyzing token lengths by message role...")
    
    user_lengths = []
    assistant_lengths = []
    
    for example in dataset:
        text = example['text']
        
        # Split by message boundaries
        parts = text.split('<|im_start|>')
        
        for part in parts:
            if part.strip():
                if part.startswith('user\n'):
                    content = part.replace('user\n', '').replace('<|im_end|>', '').strip()
                    if content:
                        tokens = tokenizer.encode(content, add_special_tokens=False)
                        user_lengths.append(len(tokens))
                elif part.startswith('assistant\n'):
                    content = part.replace('assistant\n', '').replace('<|im_end|>', '').strip()
                    if content:
                        tokens = tokenizer.encode(content, add_special_tokens=False)
                        assistant_lengths.append(len(tokens))
    
    print(f"\nUser messages:")
    if user_lengths:
        print(f"  Count: {len(user_lengths)}")
        print(f"  Mean: {np.mean(user_lengths):.2f} tokens")
        print(f"  Median: {np.median(user_lengths):.2f} tokens")
        print(f"  95th percentile: {np.percentile(user_lengths, 95):.1f} tokens")
    
    print(f"\nAssistant messages:")
    if assistant_lengths:
        print(f"  Count: {len(assistant_lengths)}")
        print(f"  Mean: {np.mean(assistant_lengths):.2f} tokens")
        print(f"  Median: {np.median(assistant_lengths):.2f} tokens")
        print(f"  95th percentile: {np.percentile(assistant_lengths, 95):.1f} tokens")

def filter_and_upload_dataset(original_dataset_name, new_dataset_name, max_tokens=10000):
    """
    Filter dataset to remove examples with more than max_tokens and upload to HuggingFace Hub
    
    Args:
        original_dataset_name: Name of the original dataset on HuggingFace Hub
        new_dataset_name: Name for the new filtered dataset
        max_tokens: Maximum number of tokens allowed per example
    """
    print(f"Loading original dataset: {original_dataset_name}")
    dataset = load_dataset(original_dataset_name)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-0.5B-bnb-4bit")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Process each split in the dataset
    filtered_splits = {}
    
    for split_name, split_data in dataset.items():
        print(f"\nProcessing {split_name} split...")
        print(f"Original size: {len(split_data)} examples")
        
        # Format the dataset for SFT to get consistent text format
        formatted_split = split_data.map(format_dataset_for_sft, remove_columns=split_data.column_names)
        
        # Filter examples based on token length
        filtered_examples = []
        removed_count = 0
        
        for i, example in enumerate(formatted_split):
            if i % 1000 == 0:
                print(f"Processed {i}/{len(formatted_split)} examples")
            
            text = example['text']
            tokens = tokenizer.encode(text, add_special_tokens=True)
            
            if len(tokens) <= max_tokens:
                # Convert back to original format by recreating the messages
                filtered_examples.append(split_data[i])
            else:
                removed_count += 1
        
        print(f"Filtered size: {len(filtered_examples)} examples")
        print(f"Removed: {removed_count} examples ({removed_count/len(split_data)*100:.1f}%)")
        
        # Create new dataset split
        if filtered_examples:
            filtered_splits[split_name] = Dataset.from_list(filtered_examples)
        else:
            print(f"Warning: No examples remain in {split_name} split after filtering")
    
    # Create DatasetDict
    if filtered_splits:
        filtered_dataset = DatasetDict(filtered_splits)
        
        print(f"\nUploading filtered dataset to: {new_dataset_name}")
        try:
            # Try to login (will prompt for token if needed)
            # Note: User should have HF_TOKEN environment variable set or run `huggingface-cli login`
            filtered_dataset.push_to_hub(new_dataset_name)
            print(f"Successfully uploaded dataset to {new_dataset_name}")
            
            # Print summary
            print(f"\nDataset Summary:")
            for split_name, split_data in filtered_dataset.items():
                print(f"  {split_name}: {len(split_data)} examples")
                
        except Exception as e:
            print(f"Error uploading dataset: {e}")
            print("Make sure you're logged in to HuggingFace Hub:")
            print("  1. Run: huggingface-cli login")
            print("  2. Or set HF_TOKEN environment variable")
            print("  3. Or ensure you have write permissions to the dataset name")
            
            # Save locally as backup
            print("Saving filtered dataset locally as backup...")
            os.makedirs('./filtered_dataset', exist_ok=True)
            filtered_dataset.save_to_disk('./filtered_dataset')
            print("Saved to ./filtered_dataset/")
    else:
        print("No data remains after filtering. No dataset created.")

def create_train_test_split_dataset(original_dataset_name, new_dataset_name, max_tokens=10000, train_ratio=0.8, random_seed=42):
    """
    Create a new dataset with 80/20 train/test split and optional token filtering
    
    Args:
        original_dataset_name: Name of the original dataset on HuggingFace Hub
        new_dataset_name: Name for the new dataset with train/test split
        max_tokens: Maximum number of tokens allowed per example (None for no filtering)
        train_ratio: Ratio for training split (default 0.8 for 80/20 split)
        random_seed: Random seed for reproducible splits
    """
    print(f"Loading original dataset: {original_dataset_name}")
    dataset = load_dataset(original_dataset_name)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-0.5B-bnb-4bit")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Combine all splits into one dataset for reshuffling
    all_examples = []
    
    for split_name, split_data in dataset.items():
        print(f"Processing {split_name} split with {len(split_data)} examples...")
        
        for example in split_data:
            all_examples.append(example)
    
    print(f"Total examples before filtering: {len(all_examples)}")
    
    # Filter examples based on token length if max_tokens is specified
    if max_tokens:
        print(f"Filtering examples with more than {max_tokens} tokens...")
        filtered_examples = []
        removed_count = 0
        
        for i, example in enumerate(all_examples):
            if i % 1000 == 0:
                print(f"Processed {i}/{len(all_examples)} examples")
            
            # Format to get text representation
            formatted_example = format_dataset_for_sft(example)
            text = formatted_example['text']
            tokens = tokenizer.encode(text, add_special_tokens=True)
            
            if len(tokens) <= max_tokens:
                filtered_examples.append(example)
            else:
                removed_count += 1
        
        print(f"Removed {removed_count} examples ({removed_count/len(all_examples)*100:.1f}%)")
        all_examples = filtered_examples
    
    print(f"Total examples after filtering: {len(all_examples)}")
    
    if len(all_examples) == 0:
        print("No examples remain after filtering. Aborting.")
        return
    
    # Create train/test split
    print(f"Creating {int(train_ratio*100)}/{int((1-train_ratio)*100)} train/test split...")
    
    # Use sklearn's train_test_split for random splitting
    train_examples, test_examples = train_test_split(
        all_examples, 
        train_size=train_ratio, 
        random_state=random_seed,
        shuffle=True
    )
    
    print(f"Train examples: {len(train_examples)}")
    print(f"Test examples: {len(test_examples)}")
    
    # Create datasets
    train_dataset = Dataset.from_list(train_examples)
    test_dataset = Dataset.from_list(test_examples)
    
    # Create DatasetDict with train/test splits
    split_dataset = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    
    print(f"\nUploading dataset with train/test split to: {new_dataset_name}")
    try:
        split_dataset.push_to_hub(new_dataset_name)
        print(f"Successfully uploaded dataset to {new_dataset_name}")
        
        # Print summary
        print(f"\nDataset Summary:")
        print(f"  train: {len(train_dataset)} examples")
        print(f"  test: {len(test_dataset)} examples")
        print(f"  total: {len(train_dataset) + len(test_dataset)} examples")
        
        # Analyze token lengths for each split
        print(f"\nAnalyzing token lengths for each split...")
        
        for split_name, split_data in split_dataset.items():
            print(f"\n{split_name.upper()} SPLIT:")
            formatted_split = split_data.map(format_dataset_for_sft, remove_columns=split_data.column_names)
            stats, token_lengths = analyze_token_lengths(formatted_split, tokenizer, sample_size=min(1000, len(formatted_split)))
            
            print(f"  Examples: {stats['count']}")
            print(f"  Mean tokens: {stats['mean']:.2f}")
            print(f"  Median tokens: {stats['median']:.2f}")
            print(f"  95th percentile: {stats['percentiles']['95th']:.1f}")
        
    except Exception as e:
        print(f"Error uploading dataset: {e}")
        print("Make sure you're logged in to HuggingFace Hub:")
        print("  1. Run: huggingface-cli login")
        print("  2. Or set HF_TOKEN environment variable")
        print("  3. Or ensure you have write permissions to the dataset name")
        
        # Save locally as backup
        print("Saving split dataset locally as backup...")
        os.makedirs('./split_dataset', exist_ok=True)
        split_dataset.save_to_disk('./split_dataset')
        print("Saved to ./split_dataset/")

def main():
    """
    Main function to run token length analysis and optionally create filtered dataset
    """
    # Option to create filtered dataset
    create_filtered_dataset = input("Do you want to create a filtered dataset (max 10k tokens)? (y/n): ").lower().strip() == 'y'
    
    # Option to create train/test split dataset
    create_split_dataset = input("Do you want to create a dataset with 80/20 train/test split? (y/n): ").lower().strip() == 'y'
    
    if create_filtered_dataset:
        print("\n" + "="*60)
        print("CREATING FILTERED DATASET")
        print("="*60)
        filter_and_upload_dataset(
            original_dataset_name="bcsandlund/arc-agi-prompts",
            new_dataset_name="bcsandlund/arc-agi-prompts-max-10k-tokens",
            max_tokens=10000
        )
        print("\nFiltered dataset creation complete!")
    
    if create_split_dataset:
        print("\n" + "="*60)
        print("CREATING TRAIN/TEST SPLIT DATASET")
        print("="*60)
        create_train_test_split_dataset(
            original_dataset_name="bcsandlund/arc-agi-prompts",
            new_dataset_name="bcsandlund/arc-agi-prompts-train-test-split",
            max_tokens=10000,  # Set to None if you don't want token filtering
            train_ratio=0.8,
            random_seed=42
        )
        print("\nTrain/test split dataset creation complete!")
    
    if not (create_filtered_dataset or create_split_dataset):
        # Ask if user wants to continue with analysis
        continue_analysis = input("\nDo you want to continue with token length analysis? (y/n): ").lower().strip() == 'y'
        if not continue_analysis:
            return
    else:
        # Ask if user wants to continue with analysis
        continue_analysis = input("\nDo you want to continue with token length analysis? (y/n): ").lower().strip() == 'y'
        if not continue_analysis:
            return

    print("\n" + "="*60)
    print("TOKEN LENGTH ANALYSIS")
    print("="*60)
    
    print("Loading dataset...")
    dataset = load_dataset("bcsandlund/arc-agi-prompts")['train']
    
    print("Formatting dataset...")
    formatted_dataset = dataset.map(format_dataset_for_sft, remove_columns=dataset.column_names)
    
    print("Loading tokenizer...")
    # Use the same tokenizer as in the training script
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-0.5B-bnb-4bit")
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Dataset size: {len(formatted_dataset)} examples")
    
    # Analyze token lengths
    stats, token_lengths = analyze_token_lengths(formatted_dataset, tokenizer)
    
    # Print detailed statistics
    print_detailed_stats(stats)
    
    # Analyze by message role
    analyze_by_message_role(formatted_dataset, tokenizer)
    
    # Save statistics to file
    print("\nSaving results...")
    os.makedirs('./token_analysis', exist_ok=True)
    
    results_df = pd.DataFrame({
        'token_length': token_lengths
    })
    results_df.to_csv('./token_analysis/token_lengths.csv', index=False)
    
    # Save summary statistics
    with open('./token_analysis/summary_stats.txt', 'w') as f:
        f.write("TOKEN LENGTH ANALYSIS SUMMARY\n")
        f.write("="*40 + "\n\n")
        f.write(f"Total examples: {stats['count']:,}\n")
        f.write(f"Mean length: {stats['mean']:.2f} tokens\n")
        f.write(f"Median length: {stats['median']:.2f} tokens\n")
        f.write(f"Standard deviation: {stats['std']:.2f} tokens\n")
        f.write(f"Min length: {stats['min']} tokens\n")
        f.write(f"Max length: {stats['max']} tokens\n\n")
        
        f.write("PERCENTILES:\n")
        for percentile, value in stats['percentiles'].items():
            f.write(f"{percentile} percentile: {value:.1f} tokens\n")
    
    print("Analysis complete! Results saved to ./token_analysis/")

def create_filtered_dataset_direct():
    """
    Direct function to create the filtered dataset without interactive prompts
    """
    print("Creating filtered dataset (max 10k tokens)...")
    filter_and_upload_dataset(
        original_dataset_name="bcsandlund/arc-agi-prompts",
        new_dataset_name="bcsandlund/arc-agi-prompts-max-10k-tokens",
        max_tokens=10000
    )

def create_train_test_split_direct():
    """
    Direct function to create the train/test split dataset without interactive prompts
    """
    print("Creating train/test split dataset (80/20 split, max 10k tokens)...")
    create_train_test_split_dataset(
        original_dataset_name="bcsandlund/arc-agi-prompts",
        new_dataset_name="bcsandlund/arc-agi-prompts-train-test-split",
        max_tokens=10000,
        train_ratio=0.8,
        random_seed=42
    )

if __name__ == "__main__":
    # Uncomment one of the lines below to directly create datasets without prompts
    # create_filtered_dataset_direct()
    # create_train_test_split_direct()
    
    # Or run the full interactive analysis
    main() 