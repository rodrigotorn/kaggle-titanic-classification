import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

from src.base_stage import BaseStage


class Preprocess(BaseStage):

  def fill_nan(self, series: pd.Series) -> pd.Series:
    if series.dtype is np.dtype(object):
      return series.fillna(series.mode())
    else:
      return series.fillna(series.mean())
  
  def transform(
    self,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    preprocessed: List = []
    for df in [train_df, test_df]:
      df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
      df = df.apply(self.fill_nan, axis=0)

      pclass_dummy = pd.get_dummies(df['Pclass'])
      pclass_dummy.rename(columns={1:'Pclass1', 2:'Pclass2', 3:'Pclass3'}, inplace=True)
      df = pd.concat([df, pclass_dummy], axis=1)
      df.drop('Pclass', axis = 1, inplace=True)

      sex_dummy = pd.get_dummies(df['Sex'])
      df = pd.concat([df, sex_dummy], axis=1)
      df.drop('Sex', axis = 1, inplace=True)
      preprocessed.append(df)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(preprocessed[0])
    X_test = scaler.transform(preprocessed[1])

    return X_train, X_test
