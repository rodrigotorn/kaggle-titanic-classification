"""Concrete Random Forest model"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from src._model import BaseModel

def grid_search():
  params = [{'n_estimators': [50, 100, 200, 500, 1000]}]
  
  return GridSearchCV(
    RandomForestClassifier(),
    param_grid=params,
    scoring='accuracy',
    cv=5
  )


class RF(BaseModel):
  """Concrete Random Forest model"""
  def __init__(
    self,
    search_params=False,
  ) -> None:
    if search_params:
      self._rf = grid_search()
    else:
      self._rf = RandomForestClassifier()

  def train(
    self,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
  ) -> np.ndarray:
    self._rf.fit(x_train, y_train)
    return cross_val_score(
      self._rf,
      x_train,
      y_train,
      cv=5,
    ).mean()

  def predict(
    self,
    x_test: pd.DataFrame,
  ) -> pd.Series:
    return self._rf.predict(x_test)
