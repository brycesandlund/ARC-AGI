from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
from typing import Generator, List, Type, TypeVar
import copy

from enums import ModelType

# Set a seed for reproducibility
random.seed(42)

T = TypeVar("T", bound="ProblemInstance")


@dataclass
class ProblemInstance:
    prompt: str

    @classmethod
    def from_instance(cls: Type[T], instance: T, new_prompt: str) -> T:
        new_instance = copy.deepcopy(instance)
        new_instance.prompt = new_prompt
        return new_instance


class Dataset(ABC):
    @abstractmethod
    def generate_math_problems(
        self, dataset_size: int, model_type: ModelType
    ) -> Generator[ProblemInstance, None, None]:
        """
        Generator function that creates math problems.
        Yields ProblemInstance object.
        """
        ...

    @abstractmethod
    def math_reward_func(
        self, completions: List[str], problem_instances: List[ProblemInstance], model_type: ModelType
    ) -> List[float]:
        """
        Reward function that evaluates mathematical correctness.
        """
        ...
