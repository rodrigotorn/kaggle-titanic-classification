"""Concrete Multi-layer Perceptron model"""
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from src._model import BaseModel
import logging


logger = logging.getLogger(__name__)


class MLP(BaseModel):
  """Concrete Multi-layer Perceptron model"""
  def __init__(self) -> None:
    pass

  def predict(
    self,
    x_train: np.ndarray,
    y_train: pd.DataFrame,
    x_test: np.ndarray,
  ) -> pd.Series:
    model = MLPClassifier(
      random_state=3,
      hidden_layer_sizes=(11,)
    )
    logger.info('Predicting with MLP model')
    model.fit(x_train, y_train)
    return model.predict(x_test)
