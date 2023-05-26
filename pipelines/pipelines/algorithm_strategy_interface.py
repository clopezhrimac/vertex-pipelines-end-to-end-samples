import json
from abc import ABCMeta, abstractmethod
from typing import List
from dataclasses import dataclass


@dataclass
class AlgorithmStrategyInterface(metaclass=ABCMeta):
    @property
    @abstractmethod
    def model_slug(self) -> str:
        pass

    @abstractmethod
    def hyperparameters(self, **kwargs) -> dict:
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

    def to_dict(self, **kwargs):
        hparams = self.hyperparameters(**kwargs)
        return json.dumps(
            {
                "model_slug": self.model_slug,
                "hyperparameters": hparams,
                "train_container_uri": self.train_container_uri,
                "serve_container_uri": self.serve_container_uri,
                "requirements": self.requirements,
            }
        )
