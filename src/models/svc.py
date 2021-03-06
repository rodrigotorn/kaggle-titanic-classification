"""Concrete SVC model"""
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from src._model import BaseModel
import logging


logger = logging.getLogger(__name__)


class SVC(BaseModel):
  """Concrete SVC model"""
  def __init__(self) -> None:
    pass

  def predict(
    self,
    x_train: np.ndarray,
    y_train: pd.DataFrame,
    x_test: np.ndarray,
  ) -> pd.Series:
    model = SVC(
      random_state=3,
      kernel='rbf',
    )
    logger.info('Predicting with SVC model')
    model.fit(x_train, y_train)
    return model.predict(x_test)
