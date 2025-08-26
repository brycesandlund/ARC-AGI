from enum import Enum

class ModelType(Enum):
    BASE = "base"
    THINKING = "thinking"
    INSTRUCT = "instruct"

    def __str__(self):
        return self.value

class EvaluationResult(Enum):
    INVALID_EXPRESSION = "Invalid expression"
    INCORRECT_RESULT = "Valid expression, incorrect result"
    CORRECT_RESULT = "Correct result"

    def __str__(self):
        return self.value
