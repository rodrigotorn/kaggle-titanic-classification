"""Concrete Multi-layer Perceptron model"""
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from src._model import BaseModel, grid_search


class MLP(BaseModel):
  """Concrete Multi-layer Perceptron model"""
  def __init__(
    self,
    search_params=False,
    params=None,
  ) -> None:
    if search_params:
      self._mlp = grid_search(MLPClassifier(), params)
    else:
      self._mlp = MLPClassifier(
        hidden_layer_sizes=(7,),
        max_iter=500,
        activation='relu',
        solver='adam',
        random_state=3
      )

  def train(
    self,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
  ) -> np.ndarray:
    self._mlp.fit(x_train, y_train)
    return cross_val_score(
      self._mlp,
      x_train,
      y_train,
      cv=5,
    ).mean()

  def predict(
    self,
    x_test: pd.DataFrame,
  ) -> pd.Series:
    return self._mlp.predict(x_test)
