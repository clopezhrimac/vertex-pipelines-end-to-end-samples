from abc import ABCMeta, abstractmethod
from typing import List
from dataclasses import dataclass


@dataclass
class AlgorithmStrategyInterface(metaclass=ABCMeta):
    @property
    @abstractmethod
    def train_script_name(self):
        pass

    @property
    @abstractmethod
    def table_suffix(self) -> str:
        pass

    @property
    @abstractmethod
    def hyperparameters(self, **kwargs) -> dict:
        pass

    @property
    @abstractmethod
    def default_model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def default_table_suffix(self) -> str:
        pass

    @property
    @abstractmethod
    def script_filename(self) -> str:
        pass

    @property
    @abstractmethod
    def train_container_uri(self) -> str:
        pass

    @property
    @abstractmethod
    def serve_container_uri(self) -> str:
        pass

    @property
    @abstractmethod
    def requirements(self) -> List[str]:
        pass
