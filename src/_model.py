"""Abstract base class for ML models"""

from abc import ABC, abstractmethod


class BaseModel(ABC):

  @abstractmethod
  def predict(self):
    pass
