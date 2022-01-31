"""Concrete Adaboost model"""
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from src._model import BaseModel


class Adaboost(BaseModel):
  """Concrete Adaboost model"""
  def __init__(self) -> None:
    pass

  def predict(
    self,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
  ) -> pd.Series:
    model = AdaBoostClassifier(
      random_state=3,
      n_estimators=2,
    )
    model.fit(x_train, y_train)
    return model.predict(x_test)
