import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
from datasets import load_from_disk

# Set up consistent color mapping
def create_color_map():
    """Create a consistent color map for ARC tasks (0-9)"""
    # Define a bright, distinct color palette
    colors = [
        '#FFFFFF',  # 0: White
        '#FF0000',  # 1: Red
        '#00FF00',  # 2: Green
        '#0000FF',  # 3: Blue
        '#FFFF00',  # 4: Yellow
        '#FF00FF',  # 5: Magenta
        '#00FFFF',  # 6: Cyan
        '#FFA500',  # 7: Orange
        '#800080',  # 8: Purple
        '#A52A2A',  # 9: Brown
    ]
    return mcolors.ListedColormap(colors)

def plot_grid(grid, ax=None, title=None):
    """Plot a single grid with colors"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    
    # Convert to numpy array if it's a list
    if isinstance(grid, list):
        grid = np.array(grid)
    
    # Handle 3D arrays (squeeze first dimension if it's 1)
    if len(grid.shape) == 3 and grid.shape[0] == 1:
        grid = np.squeeze(grid, axis=0)
    
    # Create the color map
    cmap = create_color_map()
    
    # Plot the grid
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=9)
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add numbers in cells
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, str(grid[i, j]), ha='center', va='center', 
                   color='black' if grid[i, j] < 5 else 'white', fontsize=8)
    
    if title:
        ax.set_title(title)
    
    return ax

def visualize_example(example):
    """Visualize a single ARC example with all training pairs and test input/output"""
    # Get data from the example
    train_inputs = example.get("raw_train_inputs", [])
    train_outputs = example.get("raw_train_outputs", [])
    test_input = example.get("raw_test_inputs", [None])[0]
    test_output = example.get("raw_solution", None)
    example_id = example.get("id", "unknown")
    
    # Determine the number of training examples
    n_train = len(train_inputs)
    
    # Create a figure with subplots for each training pair and test pair
    fig = plt.figure(figsize=(15, 3 + 3 * n_train))
    
    # Add title with example ID
    fig.suptitle(f"ARC Example: {example_id}", fontsize=16)
    
    # Plot training examples (input-output pairs)
    for i in range(n_train):
        # Input grid
        ax1 = plt.subplot(n_train + 1, 2, i*2 + 1)
        plot_grid(train_inputs[i], ax=ax1, title=f"Training Input {i+1}")
        
        # Output grid
        ax2 = plt.subplot(n_train + 1, 2, i*2 + 2)
        plot_grid(train_outputs[i], ax=ax2, title=f"Training Output {i+1}")
    
    # Plot test example
    ax_test_in = plt.subplot(n_train + 1, 2, n_train*2 + 1)
    plot_grid(test_input, ax=ax_test_in, title="Test Input")
    
    if test_output is not None:
        ax_test_out = plt.subplot(n_train + 1, 2, n_train*2 + 2)
        plot_grid(test_output, ax=ax_test_out, title="Expected Output")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the title
    return fig

def browse_dataset(dataset_path='arc_small_input_output_hf_split/SFT', start_idx=0, max_examples=5):
    """Load the dataset and visualize multiple examples"""
    # Load the dataset
    print(f"Loading dataset from {dataset_path}")
    dataset = load_from_disk(dataset_path)
    print(f"Dataset loaded with {len(dataset)} examples")
    
    # Create output directory for saved visualizations
    os.makedirs("visualizations", exist_ok=True)
    
    # Visualize examples
    end_idx = min(start_idx + max_examples, len(dataset))
    for i in range(start_idx, end_idx):
        example = dataset[i]
        example_id = example.get("id", f"example_{i}")
        
        print(f"Visualizing example {example_id} ({i+1}/{end_idx})")
        
        # Create visualization
        fig = visualize_example(example)
        
        # Save the figure
        fig.savefig(f"visualizations/{example_id}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved visualization to visualizations/{example_id}.png")

def main():
    parser = argparse.ArgumentParser(description='Visualize ARC dataset examples')
    parser.add_argument('--dataset_path', type=str, default='arc_small_input_output_hf_split/SFT', 
                        help='Path to the dataset')
    parser.add_argument('--start_idx', type=int, default=0, 
                        help='Starting index for examples to visualize')
    parser.add_argument('--max_examples', type=int, default=5, 
                        help='Maximum number of examples to visualize')
    parser.add_argument('--example_id', type=str, default=None,
                        help='Visualize a specific example by ID')
    args = parser.parse_args()
    
    # Load the dataset
    dataset = load_from_disk(args.dataset_path)
    
    if args.example_id:
        # Find and visualize a specific example by ID
        for i, example in enumerate(dataset):
            if example.get("id") == args.example_id:
                print(f"Visualizing example {args.example_id}")
                fig = visualize_example(example)
                fig.savefig(f"visualizations/{args.example_id}.png", dpi=150, bbox_inches='tight')
                plt.show()
                print(f"Saved visualization to visualizations/{args.example_id}.png")
                break
        else:
            print(f"Example with ID {args.example_id} not found.")
    else:
        # Browse through multiple examples
        browse_dataset(args.dataset_path, args.start_idx, args.max_examples)

if __name__ == "__main__":
    main() 