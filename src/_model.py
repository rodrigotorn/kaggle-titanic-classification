"""Abstract base class for ML models"""

from abc import ABC, abstractmethod
from sklearn.model_selection import GridSearchCV


class BaseModel(ABC):

  @abstractmethod
  def predict(self):
    pass
