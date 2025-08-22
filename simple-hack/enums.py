from enum import Enum

class ModelType(Enum):
    BASE = "base"
    THINKING = "thinking"
    INSTRUCT = "instruct"

    def __str__(self):
        return self.value
