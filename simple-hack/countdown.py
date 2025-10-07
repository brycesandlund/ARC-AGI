import random
from dataclasses import dataclass
from typing import Generator, List

from dataset import Dataset, ProblemInstance
from enums import EvaluationResult, ModelType
from functions import parse_completion

ZERO_REWARD_LOG_FREQUENCY = 0.1
ONE_REWARD_LOG_FREQUENCY = 0.2
POINT_TWO_REWARD_LOG_FREQUENCY = 1
POINT_ZERO_FIVE_REWARD_LOG_FREQUENCY = 0.1


@dataclass
class CountdownProblemInstance(ProblemInstance):
    target: int
    numbers: List[int]


class CountdownDataset(Dataset):
    def _is_correct(self, content, target):
        """
        Evaluates the mathematical expression in content and returns an EvaluationResult enum.
        
        Args:
            content (str): The mathematical expression to evaluate
            target (float): The target value the expression should equal
            
        Returns:
            EvaluationResult: An enum indicating if the expression is correct, incorrect, or invalid.
        """
        try:
            # Clean the content by stripping whitespace and newlines
            expression = content.strip().replace("×", "*").replace("÷", "/").replace("−", "-")

            # Evaluate the mathematical expression
            result = eval(expression)

            # Check if the result equals the target (with some floating point tolerance)
            if abs(result - target) < 1e-10:
                return EvaluationResult.CORRECT_RESULT
            else:
                return EvaluationResult.INCORRECT_RESULT

        except (SyntaxError, NameError, ZeroDivisionError, TypeError, ValueError) as e:
            # Return False if the expression is invalid or causes an error
            # print(f"Error evaluating expression '{content}': {e}")
            return EvaluationResult.INVALID_EXPRESSION

    def _decompose_target(self, target, num_count=4, num_range=10):
        """
        Decomposes a target number into num_count numbers using +, -, *, / operators.
        
        Args:
            target (float): The target number to decompose
            num_count (int): Number of numbers to generate (default 4)
            num_range (int): Range for random numbers (1 to num_range)
            
        Returns:
            tuple: (list of numbers, expression string) that can be combined with operators to equal target
        """
        if num_count == 1:
            if abs(target - round(target)) < 1e-10:
                return [round(target)], str(round(target))
            return [target], str(target)

        # Pick a random number and operator
        rand_num = random.randint(1, num_range)
        operator = random.choice(["+", "-", "*", "/"])

        # Calculate what the previous result needs to be based on inverse operation
        try:
            if operator == "+":
                # If we want prev_result + rand_num = target, then prev_result = target - rand_num
                prev_result = target - rand_num
            elif operator == "-":
                # If we want prev_result - rand_num = target, then prev_result = target + rand_num
                prev_result = target + rand_num
            elif operator == "*":
                # If we want prev_result * rand_num = target, then prev_result = target / rand_num
                if rand_num == 0:
                    # Avoid division by zero
                    return self._decompose_target(target, num_count, num_range)
                prev_result = target / rand_num
            elif operator == "/":
                # If we want prev_result / rand_num = target, then prev_result = target * rand_num
                prev_result = target * rand_num

            # Recurse to get the remaining numbers and expression
            remaining_numbers, prev_expression = self._decompose_target(prev_result, num_count - 1, num_range)

            # Build the complete expression
            # Add parentheses for clarity when needed
            if num_count > 2 and operator in ["*", "/"] and any(op in prev_expression for op in ["+", "-"]):
                expression = f"({prev_expression}) {operator} {rand_num}"
            else:
                expression = f"{prev_expression} {operator} {rand_num}"

            # Add our random number to the list
            remaining_numbers.append(rand_num)
            return remaining_numbers, expression

        except (ZeroDivisionError, OverflowError):
            # If we get an invalid operation, try again
            return self._decompose_target(target, num_count, num_range)

    def _generate_problem(self, target, num_count=4, num_range=10):
        """
        Generates a problem by sampling from decompose_target until all numbers are integers within 1..num_range.
        
        Args:
            target (float): The target number to decompose
            num_count (int): Number of numbers to generate (default 4)
            num_range (int): Range for random numbers (1 to num_range)
            
        Returns:
            tuple: (list of numbers, expression string) where all numbers are integers in [1, num_range]
        """
        max_attempts = 1000  # Prevent infinite loops

        for attempt in range(max_attempts):
            numbers, expression = self._decompose_target(target, num_count, num_range)

            # Check if all numbers are integers and within range
            if (
                numbers
                and all(isinstance(num, int) or (isinstance(num, float) and num.is_integer()) for num in numbers)
                and all(1 <= int(num) <= num_range for num in numbers)
            ):
                # Convert all numbers to integers for clean output
                int_numbers = [int(num) for num in numbers]
                random.shuffle(int_numbers)
                return int_numbers, expression

        # If we couldn't find a valid solution after max_attempts, return None
        print(f"Warning: Could not generate valid problem for target {target} after {max_attempts} attempts")
        return None, None

    def _extract_numbers_from_expression(self, expression):
        """
        Extracts all numbers from a mathematical expression.
        
        Args:
            expression (str): Mathematical expression
            
        Returns:
            list: List of numbers found in the expression
        """
        # Remove all non-digit, non-decimal point, and non-space characters
        # Keep only numbers and spaces
        import re

        # Find all numbers (integers and decimals) in the expression
        numbers = re.findall(r"\d+(?:\.\d+)?", expression)

        # Convert to integers (assuming we're working with integers based on the problem setup)
        return [int(float(num)) for num in numbers]

    def _numbers_match(self, prompt_numbers, expression_numbers):
        """
        Checks if the numbers used in the expression exactly match those specified in the prompt.
        
        Args:
            prompt_numbers (list): Numbers specified in the prompt
            expression_numbers (list): Numbers extracted from the expression
            
        Returns:
            bool: True if numbers match exactly (same numbers, same frequency)
        """
        # Sort both lists and compare
        return sorted(prompt_numbers) == sorted(expression_numbers)

    def generate_math_problems(
        self, dataset_size: int, model_type: ModelType
    ) -> Generator[ProblemInstance, None, None]:
        """
        Generator function that creates math problems.
        Yields ProblemInstance object.
        """

        for _ in range(dataset_size):
            target = random.randint(0, 50)
            num_count = random.choice([4])
            numbers, expression = self._generate_problem(target, num_count=num_count, num_range=10)

            if numbers and expression:  # Only yield if we successfully generated a problem
                # Create prompt similar to test_inference
                numbers_str = ", ".join(map(str, numbers))

                if model_type == ModelType.BASE:
                    prompt_content = (
                        f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
                        "The assistant first thinks about the reasoning process in the <reasoning></reasoning> tags and then provides the user with the answer.\n"
                        f"User: Using the numbers {numbers_str} exactly once in mathematical notation using addition, subtraction, multiplication, division, and/or parentheses, create an expression that equals {target}. Show your "
                        "work in the <reasoning> </reasoning> tags, then output a single final expression with no additional text or notation. For example, <reasoning>...</reasoning>10+8-2.\n"
                        "Assistant: Let me solve this step by step."
                    )

                elif model_type == ModelType.INSTRUCT:
                    # INSTRUCT-TUNED PROMPT (no thinking):
                    prompt_content = f"Using the numbers {numbers_str} exactly once, create a mathematical expression using +, -, *, /, and/or () that equals {target}. Please reason step by step, and put your final expression in <answer></answer> tags, for example, <answer>4*5-4</answer>."

                    r = random.random()
                    if r < 0.7:
                        prompt_content = "This is a longer prompt. " + prompt_content
                    
                    # prompt_content = f"Using the numbers {numbers_str} exactly once, create a mathematical expression using +, -, *, /, and/or () that equals {target}. Please reason step by step, and put your final expression after #### with no additional text, for example, ####4*5-4."

                else:  # Catches ModelType.THINKING
                    # REASONING-TRAINED PROMPT:
                    prompt_content = f"Using the numbers {numbers_str} exactly once in mathematical notation using addition, subtraction, multiplication, division, and/or parentheses, create an expression that equals {target}. Keep your reasoning in the <think> block brief. Answer exactly in plain mathematical notation (DO NOT USE LATEX), WITH NO ADDITIONAL TEXT. For example, if the provided numbers are 8, 3, 2, 3, a valid answer would be: (3 / 3 + 2) * 8. Or, if the numbers were 8, 2, 9, 9, a valid answer would be 9 + 9 - 2 + 8. ANSWER AS SOON AS A CORRECT EXPRESSION IS FOUND. Do not include = {target} in your answer."

                yield CountdownProblemInstance(prompt=prompt_content, target=target, numbers=numbers)

    def math_reward_func(
        self, completions: List[str], problem_instances: List[ProblemInstance], model_type: ModelType
    ) -> List[float]:
        """
        Reward function that evaluates mathematical correctness using is_correct.
        
        Args:
            completions: List of generated completions
            problem_instances: List of ProblemInstance objects
            model_type: The type of model
            
        Returns:
            List of reward scores (1.0 for correct, 0.0 for incorrect)
        """
        rewards = []

        for completion, problem_instance in zip(completions, problem_instances):
            target = problem_instance.target
            prompt = problem_instance.prompt
            prompt_numbers = problem_instance.numbers

            # Clean the completion and check if it's correct
            _, content = parse_completion(completion, model_type=model_type)

            reward = 0.0  # Default to 0

            # Extract numbers from expression
            expression_numbers = self._extract_numbers_from_expression(content)

            # Check both mathematical correctness and number usage
            evaluation_result = self._is_correct(content, target)
            numbers_are_correct = self._numbers_match(prompt_numbers, expression_numbers)

            # Only give full reward if both conditions are met
            if evaluation_result == EvaluationResult.CORRECT_RESULT and numbers_are_correct:
                reward = 1.0
            elif evaluation_result == EvaluationResult.CORRECT_RESULT and not numbers_are_correct:
                reward = 0.0  # Wrong numbers used
            elif evaluation_result != EvaluationResult.CORRECT_RESULT and numbers_are_correct:
                reward = 0.0  # Right numbers but wrong result
            else:
                reward = 0.0  # Both wrong

            # If the answer isn't perfect, give partial credit for formatting
            # Not had good results with partial rewards, but left here for base model.
            if reward < 1.0 and model_type == ModelType.BASE:
                start_tag, end_tag = "<reasoning>", "</reasoning>"
                num_start_tags = completion.count(start_tag)
                num_end_tags = completion.count(end_tag)

                is_well_formatted = False
                if num_start_tags == 1 and num_end_tags == 1:
                    start_pos = completion.find(start_tag)
                    end_pos = completion.find(end_tag)
                    if start_pos < end_pos:
                        is_well_formatted = True

                if is_well_formatted:
                    if (
                        evaluation_result == EvaluationResult.INCORRECT_RESULT
                        or evaluation_result == EvaluationResult.CORRECT_RESULT
                    ):
                        reward = 0.2
                    elif evaluation_result == EvaluationResult.INVALID_EXPRESSION:
                        reward = 0.05
                else:
                    reward = 0  # partial_reward

            log_prob = 0
            if reward == 1.0:
                log_prob = ONE_REWARD_LOG_FREQUENCY
            elif reward == 0.2:
                log_prob = POINT_TWO_REWARD_LOG_FREQUENCY
            elif reward == 0.05:
                log_prob = POINT_ZERO_FIVE_REWARD_LOG_FREQUENCY
            elif reward == 0.0:
                log_prob = ZERO_REWARD_LOG_FREQUENCY

            if random.random() < log_prob:
                print("\n-----")
                print(f"Prompt: {prompt}")
                print(f"Completion: {completion}")
                print(f"Parsed Content: {content}")
                print(f"Prompt Numbers: {prompt_numbers}")
                print(f"Expression Numbers: {self._extract_numbers_from_expression(content)}")
                print(f"Numbers Match: {self._numbers_match(prompt_numbers, self._extract_numbers_from_expression(content))}")
                print(f"Reward: {reward}")
                print("-----")

            rewards.append(reward)

        return rewards
