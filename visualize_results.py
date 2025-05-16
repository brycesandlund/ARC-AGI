import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
from datasets import load_from_disk
import re

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

def extract_claude_answer(response):
    """Extract the answer grid from Claude's response (inside <answer> tags) and convert to numpy array"""
    if not response:
        return None
    
    # Find content between <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
        
        # Convert string grid to 2D array
        grid_array = []
        for line in answer_text.strip().split('\n'):
            if line.strip():  # Skip empty lines
                # Convert line to list of integers
                row = [int(num) for num in re.findall(r'\d+', line)]
                if row:  # Only add non-empty rows
                    grid_array.append(row)
        
        return np.array(grid_array)
    
    return None

def visualize_result(result_file, dataset_path='arc_small_input_output_hf_split/SFT'):
    """Visualize a single result file with Claude's prediction and expected output"""
    # Load the result file
    with open(result_file, 'r') as f:
        result = json.load(f)
    
    example_id = result.get("id")
    claude_response = result.get("response")
    
    # Extract Claude's answer
    claude_grid = extract_claude_answer(claude_response)
    
    # Load the dataset
    dataset = load_from_disk(dataset_path)
    
    # Find the corresponding example in the dataset
    example = None
    for i, ex in enumerate(dataset):
        if ex.get("id") == example_id:
            example = ex
            break
    
    if example is None:
        print(f"Example with ID {example_id} not found in dataset.")
        return None
    
    # Get data from the example
    train_inputs = example.get("raw_train_inputs", [])
    train_outputs = example.get("raw_train_outputs", [])
    test_input = example.get("raw_test_inputs", [None])[0]
    expected_output = example.get("raw_solution", None)
    
    # Determine the number of training examples
    n_train = len(train_inputs)
    
    # Create a figure with subplots for training pairs, test input, expected output, and Claude's output
    fig = plt.figure(figsize=(15, 3 + 3 * (n_train + 1)))
    
    # Add title with example ID
    fig.suptitle(f"ARC Example: {example_id}", fontsize=16)
    
    # Plot training examples (input-output pairs)
    for i in range(n_train):
        # Input grid
        ax1 = plt.subplot(n_train + 2, 2, i*2 + 1)
        plot_grid(train_inputs[i], ax=ax1, title=f"Training Input {i+1}")
        
        # Output grid
        ax2 = plt.subplot(n_train + 2, 2, i*2 + 2)
        plot_grid(train_outputs[i], ax=ax2, title=f"Training Output {i+1}")
    
    # Plot test input
    ax_test_in = plt.subplot(n_train + 2, 2, n_train*2 + 1)
    plot_grid(test_input, ax=ax_test_in, title="Test Input")
    
    # Plot expected output
    if expected_output is not None:
        ax_expected = plt.subplot(n_train + 2, 2, n_train*2 + 2)
        plot_grid(expected_output, ax=ax_expected, title="Expected Output")
    
    # Plot Claude's output
    if claude_grid is not None:
        # Check if last row would have 2 plots or just 1
        has_expected = expected_output is not None
        
        if has_expected:
            # If expected output exists, plot Claude's output on next row
            ax_claude = plt.subplot(n_train + 2, 2, (n_train+1)*2 + 1)
        else:
            # If no expected output, plot Claude's output next to test input
            ax_claude = plt.subplot(n_train + 2, 2, n_train*2 + 2)
        
        # Determine if Claude's output is correct
        is_correct = False
        if expected_output is not None and claude_grid is not None:
            try:
                expected_np = np.array(expected_output)
                # Check dimensions and values
                is_correct = (claude_grid.shape == expected_np.shape and 
                             np.array_equal(claude_grid, expected_np))
            except:
                is_correct = False
        
        claude_title = "Claude's Output"
        if expected_output is not None:
            claude_title += " (" + ("✓ Correct" if is_correct else "✗ Incorrect") + ")"
        
        plot_grid(claude_grid, ax=ax_claude, title=claude_title)
    
    # Show Claude's reasoning
    if claude_response and has_expected and claude_grid is not None:
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', claude_response, re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            # Create a text box for reasoning
            ax_reasoning = plt.subplot(n_train + 2, 2, (n_train+1)*2 + 2)
            ax_reasoning.axis('off')
            ax_reasoning.text(0, 1, "Claude's Reasoning:", fontsize=10, fontweight='bold')
            
            # Truncate reasoning if too long
            max_chars = 500
            if len(reasoning) > max_chars:
                reasoning = reasoning[:max_chars] + "..."
            
            ax_reasoning.text(0, 0.9, reasoning, fontsize=8, wrap=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the title
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize Claude\'s results from ARC dataset')
    parser.add_argument('--results_dir', type=str, default='results', 
                        help='Directory containing result files')
    parser.add_argument('--dataset_path', type=str, default='arc_small_input_output_hf_split/SFT', 
                        help='Path to the dataset')
    parser.add_argument('--example_id', type=str, default=None,
                        help='Visualize a specific example by ID')
    args = parser.parse_args()
    
    # Create output directory for saved visualizations
    os.makedirs("result_visualizations", exist_ok=True)
    
    # If a specific example ID is provided
    if args.example_id:
        result_file = os.path.join(args.results_dir, f"{args.example_id}.json")
        if os.path.exists(result_file):
            fig = visualize_result(result_file, args.dataset_path)
            if fig:
                fig.savefig(f"result_visualizations/{args.example_id}.png", dpi=150, bbox_inches='tight')
                plt.show()
                print(f"Saved visualization to result_visualizations/{args.example_id}.png")
        else:
            print(f"Result file for example {args.example_id} not found.")
    else:
        # Process all result files
        result_files = [f for f in os.listdir(args.results_dir) if f.endswith('.json')]
        
        if not result_files:
            print(f"No result files found in {args.results_dir}")
            return
        
        print(f"Found {len(result_files)} result files")
        
        for result_file in result_files:
            example_id = os.path.splitext(result_file)[0]
            print(f"Processing result for example {example_id}")
            
            fig = visualize_result(os.path.join(args.results_dir, result_file), args.dataset_path)
            if fig:
                fig.savefig(f"result_visualizations/{example_id}.png", dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved visualization to result_visualizations/{example_id}.png")

if __name__ == "__main__":
    main() 