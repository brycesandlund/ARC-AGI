import json
from datasets import Dataset, DatasetDict, Features, Value, Sequence

if __name__ == "__main__":

    # Load the small input/output puzzles dataset from JSON
    with open('small_input_output_puzzles_dataset_unofficial.json', 'r') as f:
        small_data = json.load(f)

    # Combine train and test examples
    train_examples = small_data['train'][0]['train']
    test_examples = small_data['test'][0]['test']
    all_examples = train_examples + test_examples
    print(f"Loaded {len(train_examples)} train examples and {len(test_examples)} test examples")
    print(f"Total examples: {len(all_examples)}")

    # Print the structure of the first example
    print("\nExample structure:")
    print(json.dumps(all_examples[0], indent=2))

    # Define features for the dataset
    features = Features({
        'id': Value('string'),
        'raw_train_inputs': Sequence(Sequence(Sequence(Value('int32')))),  # Array of 2D input grids
        'raw_train_outputs': Sequence(Sequence(Sequence(Value('int32')))),  # Array of 2D output grids
        'raw_test_inputs': Sequence(Sequence(Sequence(Value('int32')))),  # Array of 2D test input grids
        'raw_solution': Sequence(Sequence(Sequence(Value('int32')))),  # Array of 2D solution grids
        'train_inputs': Sequence(Value('string')),  # Array of text input grids
        'train_outputs': Sequence(Value('string')),  # Array of text output grids
        'test_inputs': Sequence(Value('string')),  # Array of text test input grids
        'solution': Sequence(Value('string')),  # Array of text solution grids
        'messages': Sequence({  # Array of conversation messages
            'role': Value('string'),
            'content': Value('string')
        })
    })

    # Shuffle and split
    import random
    random.seed(42)
    random.shuffle(all_examples)
    
    # Calculate split sizes (60% SFT, 25% RL, 15% test)
    total_size = len(all_examples)
    sft_size = 1000
    rl_size = 700
    
    sft_examples = all_examples[:sft_size]
    rl_examples = all_examples[sft_size:sft_size + rl_size]
    test_examples = all_examples[sft_size + rl_size:]
    
    print(f"New splits - SFT: {len(sft_examples)}, RL: {len(rl_examples)}, test: {len(test_examples)}")

    # Save as Hugging Face dataset format with explicit features
    sft_ds = Dataset.from_list(sft_examples, features=features)
    rl_ds = Dataset.from_list(rl_examples, features=features)
    test_ds = Dataset.from_list(test_examples, features=features)
    ds_dict = DatasetDict({
        'SFT': sft_ds,
        'RL': rl_ds,
        'test': test_ds
    })
    ds_dict.save_to_disk('arc_small_input_output_hf_split')
    print("Saved Hugging Face dataset to arc_small_input_output_hf_split/")