import pandas as pd
from sklearn.neural_network import MLPClassifier

from src.base_stage import BaseStage


class Model(BaseStage):
  def transform(self):
    pass

  def predict(
    self,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
  ) -> pd.DataFrame:
    mlp = MLPClassifier(
      hidden_layer_sizes=(50,),
      max_iter=500,
      activation='relu',
      solver='adam',
      random_state=3
    )
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    return y_pred
