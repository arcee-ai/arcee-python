#lm.py
from abc import ABC, abstractmethod

class LM(ABC):
    """Base model class"""

    @abstractmethod
    def __init__(self, model_name: str):
        self.model_name = model_name
        pass

    @abstractmethod
    def train(self, dataset_path: str):
        pass

    @abstractmethod
    def predict(self, prompt: str):
        pass
