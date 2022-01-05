import pandas as pd
import numpy as np
from src._stage import BaseStage


def _fill_nan(series: pd.Series) -> pd.Series:
  if series.dtype is np.dtype(object):
    return series.fillna(series.mode())
  else:
    return series.fillna(series.mean())


class Preprocess(BaseStage):
  def __init__(
    self,
    train_df: pd.DataFrame,
    scaler,
  ) -> None:
    self._scaler = scaler.fit(self.clean(train_df))

  def clean(
    self,
    df: pd.DataFrame,
  ) -> pd.DataFrame:

    df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
    df = df.apply(_fill_nan, axis=0)

    pclass_dummy = pd.get_dummies(df['Pclass'])
    pclass_dummy.rename(columns={1:'Pclass1', 2:'Pclass2', 3:'Pclass3'}, inplace=True)
    df = pd.concat([df, pclass_dummy], axis=1)
    df.drop('Pclass', axis = 1, inplace=True)

    sex_dummy = pd.get_dummies(df['Sex'])
    df = pd.concat([df, sex_dummy], axis=1)
    df.drop('Sex', axis = 1, inplace=True)
    return df

  def transform(
    self,
    df: pd.DataFrame,
  ) -> pd.DataFrame:
    return self._scaler.transform(self.clean(df))
