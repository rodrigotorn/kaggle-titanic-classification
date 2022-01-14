"""Abstract base class for ML models"""

from abc import ABC, abstractmethod
from sklearn.model_selection import GridSearchCV

def grid_search(classifier, params):
  return GridSearchCV(
    classifier,
    param_grid=params,
    scoring='accuracy',
    cv=5
  )


class BaseModel(ABC):

  @abstractmethod
  def train(self):
    pass

  @abstractmethod
  def predict(self):
    pass
