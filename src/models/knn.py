"""Concrete K-Nearest Neighbors model"""
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from src._model import BaseModel

def grid_search():
  params = [{'n_neighbors': [5, 10, 15, 20, 25]}]
  
  return GridSearchCV(
    KNeighborsClassifier(),
    param_grid=params,
    scoring='accuracy',
    cv=5
  )


class KNN(BaseModel):
  """Concrete K-Nearest Neighbors model"""
  def __init__(
    self,
    search_params=False,
  ) -> None:
    if search_params:
      self._knn = grid_search()
    else:
      self._knn = KNeighborsClassifier(
        n_neighbors=13
      )

  def train(
    self,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
  ) -> np.ndarray:
    self._knn.fit(x_train, y_train)
    return cross_val_score(
      self._knn,
      x_train,
      y_train,
      cv=5,
    ).mean()

  def predict(
    self,
    x_test: pd.DataFrame,
  ) -> pd.Series:
    return self._knn.predict(x_test)
