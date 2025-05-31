import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter
import statistics

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

def main():
    """
    Main function to run the token length analysis
    """
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
    import os
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

if __name__ == "__main__":
    main() 