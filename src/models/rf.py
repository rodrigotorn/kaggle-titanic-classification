"""Concrete Random Forest model"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src._model import BaseModel
import logging


logger = logging.getLogger(__name__)


class RF(BaseModel):
  """Concrete Random Forest model"""
  def __init__(self) -> None:
    pass

  def predict(
    self,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
  ) -> pd.Series:
    model = RandomForestClassifier(
      random_state=3,
      max_depth=2
    )
    logger.info('Predicting with Random Forest model')
    model.fit(x_train, y_train)
    return model.predict(x_test)
