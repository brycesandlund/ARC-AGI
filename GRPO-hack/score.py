import numpy as np
import re
import traceback

def parse_grid(grid_str):
    """
    Parse a grid string into a numpy array, handling different formats and exceptions.
    Returns None if parsing fails.
    """
    try:
        if "think" in grid_str:
            return None
        # Check if it's just text without grid structure
        if not re.search(r'[\[\]\d,]', grid_str) or len(grid_str) < 5:
            return None
            
        # First, try to directly evaluate if it's in Python list format
        if grid_str.strip().startswith('[') and grid_str.strip().endswith(']'):
            # Clean up the string to make it evaluable
            clean_str = grid_str.replace('\n', ',').replace(',,', ',')
            # Check if it's a 2D list
            if '[[' in clean_str:
                return np.array(eval(clean_str))
            else:
                # It might be a list of rows
                rows = [eval(row.strip()) for row in grid_str.split('\n') if row.strip() and row.strip().startswith('[')]
                if rows:
                    return np.array(rows)
        
        # Otherwise parse as rows of numbers
        rows = []
        for line in grid_str.strip().split('\n'):
            # Skip lines that are likely reasoning text
            if not re.search(r'[\[\d]', line):
                continue
                
            # Extract numbers from the line
            numbers = re.findall(r'\d+', line)
            if numbers:
                rows.append([int(n) for n in numbers])
        
        if rows:
            # Check if all rows have the same length
            row_lengths = [len(r) for r in rows]
            if len(set(row_lengths)) > 1:
                # Pad shorter rows
                max_len = max(row_lengths)
                rows = [r + [0] * (max_len - len(r)) for r in rows]
            return np.array(rows)
        
        return None
    except Exception as e:
        print(f"Parse error: {e}")
        return None

def calculate_score(generated_output, expected_answer):
    """
    Calculate multiple scoring metrics for comparing generated output to expected answer.
    
    Args:
        generated_output: String or array representing the model's output
        expected_answer: 2D array representing the correct answer
    
    Returns:
        Dictionary of scores and metrics, with safe default values
    """
    # Initialize with safe default values
    scores = {
        "cell_accuracy": 0.0,
        "cell_accuracy_percent": 0.0,
        "correct_cells": 0,
        "total_cells": 0,
        "pattern_score": 0.0,
        "combined_score": 0.0,
        "dimension_match": False,
        "parsed_successfully": False
    }
    
    try:
        # Convert expected_answer to numpy array if it's not already
        if isinstance(expected_answer, list):
            expected_answer = np.array(expected_answer)
        
        # Parse generated_output if it's a string
        if isinstance(generated_output, str):
            parsed_output = parse_grid(generated_output)
            if parsed_output is None:
                scores["error"] = "Could not parse generated output as a grid"
                return scores
        else:
            parsed_output = np.array(generated_output)
        
        scores["parsed_successfully"] = True
        scores["parsed_output"] = parsed_output.tolist()
        
        # Check if shapes match - if not, return zero rewards
        if parsed_output.shape != expected_answer.shape:
            scores["error"] = f"Shape mismatch: {parsed_output.shape} vs {expected_answer.shape}"
            scores["dimension_match"] = False
            # Add a small reward for at least producing a grid
            scores["cell_accuracy"] = 0.01
            scores["cell_accuracy_percent"] = 1.0
            scores["combined_score"] = 0.01
            return scores
        
        scores["dimension_match"] = True
        
        # Calculate cell accuracy
        total_cells = expected_answer.size
        correct_cells = np.sum(parsed_output == expected_answer)
        cell_accuracy = correct_cells / total_cells if total_cells > 0 else 0
        
        # Check for pattern recognition (vertical symmetry in this case)
        # pattern_score = calculate_pattern_score(parsed_output, expected_answer)
        
        combined_score = cell_accuracy
        combined_score += 0.3 # this is for the dimension match
        
        
        # Update scores dictionary
        scores.update({
            "cell_accuracy": float(cell_accuracy),
            "cell_accuracy_percent": float(cell_accuracy * 100),
            "correct_cells": int(correct_cells),
            "total_cells": int(total_cells),
            "pattern_score": 0,
            "combined_score": float(combined_score)
        })
        
        return scores
    
    except Exception as e:
        scores["error"] = f"Error calculating score: {str(e)}"
        return scores

def calculate_pattern_score(output, answer):
    """
    Calculate a score based on how well the output captures patterns in the answer.
    This implementation focuses on vertical symmetry, but can be extended.
    
    Returns a score between 0 and 1.
    """
    try:
        # Check for vertical symmetry in the answer
        answer_symmetry = check_vertical_symmetry(answer)
        
        if answer_symmetry:
            # If the answer has vertical symmetry, check if output has some symmetry
            output_symmetry = partial_vertical_symmetry(output)
            return output_symmetry
        else:
            # For other patterns, this would need to be expanded
            # For now, default to a simple pattern score based on local similarities
            return local_pattern_similarity(output, answer)
    
    except Exception as e:
        print(f"Error in pattern scoring: {e}")
        return 0.0

def check_vertical_symmetry(grid):
    """Check if grid has perfect vertical symmetry"""
    n_cols = grid.shape[1]
    for row in grid:
        for j in range(n_cols // 2):
            if row[j] != row[n_cols - j - 1]:
                return False
    return True

def partial_vertical_symmetry(grid):
    """Calculate what percentage of the grid shows vertical symmetry"""
    symmetry_count = 0
    total_pairs = 0
    
    n_rows, n_cols = grid.shape
    for i in range(n_rows):
        for j in range(n_cols // 2):
            total_pairs += 1
            if grid[i, j] == grid[i, n_cols - j - 1]:
                symmetry_count += 1
    
    return symmetry_count / total_pairs if total_pairs > 0 else 0

def local_pattern_similarity(output, answer):
    """
    Calculate similarity based on local patterns (2x2 windows)
    This is a simple approach that could be expanded for more complex patterns
    """
    if min(output.shape) < 2:  # Too small for 2x2 windows
        return 0.0
    
    similarity_score = 0
    count = 0
    
    for i in range(output.shape[0] - 1):
        for j in range(output.shape[1] - 1):
            # Extract 2x2 windows
            output_window = output[i:i+2, j:j+2]
            answer_window = answer[i:i+2, j:j+2]
            
            # Calculate similarity (ratio of matching values)
            matching = np.sum(output_window == answer_window)
            similarity_score += matching / 4  # 4 cells in a 2x2 window
            count += 1
    
    return similarity_score / count if count > 0 else 0

# Example usage
def score_arc_solution(generated_output, expected_answer):
    """Wrapper function to score an ARC solution with better error handling"""
    try:
        scores = calculate_score(generated_output, expected_answer)
        
        print("=" * 20)
        if "error" in scores:
            print(f"Error: {scores['error']}")
        
        if scores.get("parsed_successfully", False):
            if "parsed_output" in scores:
                print("\nParsed output:")
                for row in scores["parsed_output"]:
                    print(row)
            
            print("\nExpected answer:")
            if isinstance(expected_answer, list):
                for row in expected_answer:
                    print(row)
            else:
                for row in expected_answer.tolist():
                    print(row)
        else:
            print("\nCould not parse output as grid. Original output:")
            print(generated_output[:100])
        
        print("\nScoring metrics:")
        print(f"Successfully parsed: {scores.get('parsed_successfully', False)}")
        print(f"Dimensions match: {scores.get('dimension_match', False)}")
        print(f"Cell accuracy: {scores.get('cell_accuracy_percent', 0):.2f}% ({scores.get('correct_cells', 0)}/{scores.get('total_cells', 0)} cells)")
        print(f"Combined score: {scores.get('combined_score', 0):.4f}")
        print('=' * 20)
        
        return scores['combined_score']
    
    except Exception as e:
        print(f"Unexpected error in scoring function: {str(e)}")
        traceback.print_exc()
        # Return safe default values
        # return {
        #     "error": f"Scoring failed: {str(e)}",
        #     "cell_accuracy": 0.0,
        #     "cell_accuracy_percent": 0.0,
        #     "combined_score": 0.0,
        #     "parsed_successfully": False,
        #     "dimension_match": False
        # }
        return 0

# Example
def example_usage():
    # Example with well-formatted output
    generated_output_good = '[0, 1, 7, 9, 0, 0]\n[0, 1, 1, 0, 0, 0]\n[0, 4, 0, 9, 0, 0]\n[1, 4, 1, 3, 8, 9]\n[3, 6, 2, 6, 6, 6]\n[5, 6, 5, 7, 9, 4]'
    
    # Example with text that can't be parsed as a grid
    generated_output_text = '''
    <think>
    3049
2025-05-17 02:48:45

3050
2025-05-17 02:48:45
Looking at the first output row: [0,7,7,0,0,0]. The input's first row is [0,4,4,0,0,0]. How do these relate? Maybe the output's first element is 0, which is the input's first element. The second element is 7, which is 4+3. The third is 7, which is 4+3. The fourth is 0. So maybe the pattern is that for each element in the output, it's the sum of the input's elements in the same position plus some value. For example, first element is 0, which is 0. Second is 4+3=7. Third is 4+3=7. Fourth is 0. So maybe adding 3 each time. But where does the 3 come from? Maybe it's based on the row index. For example, first output row's second element is 4 + 3 (row index 0?), but that seems arbitrary.
3051
2025-05-17 02:48:45

3052
2025-05-17 02:48:45
Alternatively, maybe the output is derived by taking the sum of the input's elements in the first two columns and then adjusting. For example, first input column sum is 14, second column sum is 16. Output first two elements sum to 0+7=7. Not matching.
3053
2025-05-17 02:48:45

3054
2025-05-17 02:48:45
This is getting frustrating. Let me try looking at the output rows for example 2. The input for example 2 is:
3055
2025-05-17 02:48:45

3056
2025-05-17 02:48:45
[2, 2, 5, 0, 0, 0]
3057
2025-05-17 02:48:45

3058
2025-05-17 02:48:45
[2, 7, 4, 0, 0, 0]
3059
2025-05-17 02:48:45

3060
2025-05-17 02:48:45
[0, 6, 1, 0, 0, 0]
3061
2025-05-17 02:48:45

3062
2025-05-17 02:48:45
[0, 0, 7, 5, 4, 2]
3063
2025-05-17 02:48:45

3064
2025-05-17 02:48:45
[0, 0, 0, 4, 7, 2]
3065
2025-05-17 02:48:45

3066
2025-05-17 02:48:45
[0, 0, 0, 1, 1, 0]
3067
2025-05-17 02:48:45

3068
2025-05-17 02:48:45
The output for example 2 is:
3069
2025-05-17 02:48:45

3070
2025-05-17 02:48:45
[0, 1, 1, 0, 0, 0]'''
    
    # Example with different dimensions
    generated_output_wrong_dim = '[0, 1]\n[0, 1]\n[0, 4]'
    
    expected_answer = [
        [0, 1, 7, 7, 1, 0], 
        [0, 1, 1, 1, 1, 0], 
        [0, 4, 0, 0, 4, 0], 
        [1, 4, 1, 1, 4, 1], 
        [3, 6, 2, 2, 6, 3], 
        [5, 6, 5, 5, 6, 5]
    ]
    
    # print("\n=== Testing with well-formatted output ===")
    scores1 = score_arc_solution(generated_output_good, expected_answer)
    
    # print("\n=== Testing with text that can't be parsed as a grid ===")
    scores2 = score_arc_solution(generated_output_text, expected_answer)
    
    # print("\n=== Testing with wrong dimensions ===")
    scores3 = score_arc_solution(generated_output_wrong_dim, expected_answer)
