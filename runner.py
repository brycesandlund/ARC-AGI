from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import numpy as np
from sklearn.model_selection import train_test_split

def analyze_array_dimensions(dataset):
    input_dims = Counter()
    output_dims = Counter()
    
    for split in dataset.keys():
        print(f"\nAnalyzing dimensions for split: {split}")
        split_data = dataset[split]
        for group in split_data:
            # group is a dict with a key like 'train' or 'test' whose value is a list of examples
            for key in group:
                for example in group[key]:
                    input_shape = (len(example['input']), len(example['input'][0]))
                    output_shape = (len(example['output']), len(example['output'][0]))
                    input_dims[input_shape] += 1
                    output_dims[output_shape] += 1
    print("\nInput array dimensions:")
    for dim, count in sorted(input_dims.items()):
        print(f"{dim}: {count} examples")
    print("\nOutput array dimensions:")
    for dim, count in sorted(output_dims.items()):
        print(f"{dim}: {count} examples")

def load_and_explore_dataset():
    # Load the dataset
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("lordspline/arc-agi")
    
    # Print dataset structure
    print("\nDataset Structure:")
    print(dataset)
    
    for split in dataset.keys():
        print(f"\n--- Exploring split: {split} ---")
        split_df = pd.DataFrame(dataset[split])
        print(f"Number of examples: {len(split_df)}")
        print(f"Columns: {split_df.columns.tolist()}")
        print("First example:")
        print(json.dumps(split_df.iloc[0].to_dict(), indent=2))
        print("Data types:")
        print(split_df.dtypes)
        print("Missing values per column:")
        print(split_df.isnull().sum())
        for col in split_df.columns:
            if split_df[col].dtype == 'object':
                print(f"Value counts for {col}:")
                print(split_df[col].value_counts().head())
    
    # Analyze array dimensions
    analyze_array_dimensions(dataset)
    
    return dataset

def filter_puzzles_by_output_size(dataset, max_size=5):
    filtered_puzzles = []
    
    for split in dataset.keys():
        split_data = dataset[split]
        for group in split_data:
            for key in group:
                for example in group[key]:
                    output_shape = (len(example['output']), len(example['output'][0]))
                    if output_shape[0] <= max_size and output_shape[1] <= max_size:
                        filtered_puzzles.append(example)
    
    return filtered_puzzles

def filter_puzzles_by_both_sizes(dataset, max_size=5):
    filtered_puzzles = []
    
    for split in dataset.keys():
        split_data = dataset[split]
        for group in split_data:
            for key in group:
                for example in group[key]:
                    input_shape = (len(example['input']), len(example['input'][0]))
                    output_shape = (len(example['output']), len(example['output'][0]))
                    if (input_shape[0] <= max_size and input_shape[1] <= max_size and
                        output_shape[0] <= max_size and output_shape[1] <= max_size):
                        filtered_puzzles.append(example)
    
    return filtered_puzzles

def create_new_dataset(filtered_puzzles, test_size=0.2, random_state=42):
    # Split the data into train and test sets
    train_puzzles, test_puzzles = train_test_split(
        filtered_puzzles, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Create the new dataset structure
    new_dataset = {
        'train': [{'train': train_puzzles}],
        'test': [{'test': test_puzzles}]
    }
    
    return new_dataset

def save_dataset(dataset, output_path='small_puzzles_dataset.json'):
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    dataset = load_and_explore_dataset()
    
    # Create dataset with small output grids
    print("\nFiltering puzzles with output size <= 5x5...")
    filtered_puzzles = filter_puzzles_by_output_size(dataset)
    print(f"Found {len(filtered_puzzles)} puzzles with output size <= 5x5")
    
    # Create new dataset with 80/20 split
    print("\nCreating new dataset with 80/20 train/test split...")
    new_dataset = create_new_dataset(filtered_puzzles)
    
    # Save the new dataset
    save_dataset(new_dataset, 'small_output_puzzles_dataset.json')
    
    # Print statistics about the new dataset
    print("\nNew dataset statistics (small output):")
    print(f"Training examples: {len(new_dataset['train'][0]['train'])}")
    print(f"Test examples: {len(new_dataset['test'][0]['test'])}")
    
    # Create dataset with both small input and output grids
    print("\nFiltering puzzles with both input and output size <= 5x5...")
    filtered_puzzles_both = filter_puzzles_by_both_sizes(dataset)
    print(f"Found {len(filtered_puzzles_both)} puzzles with both input and output size <= 5x5")
    
    # Create new dataset with 80/20 split
    print("\nCreating new dataset with 80/20 train/test split...")
    new_dataset_both = create_new_dataset(filtered_puzzles_both)
    
    # Save the new dataset
    save_dataset(new_dataset_both, 'small_input_output_puzzles_dataset.json')
    
    # Print statistics about the new dataset
    print("\nNew dataset statistics (small input and output):")
    print(f"Training examples: {len(new_dataset_both['train'][0]['train'])}")
    print(f"Test examples: {len(new_dataset_both['test'][0]['test'])}")
