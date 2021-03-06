"""Preprocess class specific for titanic dataset"""

import pandas as pd
import numpy as np
import logging


logger = logging.getLogger(__name__)


class Preprocess():
  """Preprocess class. Scaler is defined when instantiated"""
  def __init__(
    self,
    scaler=None,
  ) -> None:
    self.scaler = scaler if scaler else None


  def transform(self, train_df, test_df):
    transformed_df = []
    logger.info('Selecting only the desired columns')
    for df in [train_df, test_df]:
      df = df[[
        'Pclass',
        'Sex',
        'Age',
        'SibSp',
        'Parch',
        'Fare',
        'Embarked'
      ]]

      logger.info('Filling gaps from Age and Embarked columns')
      df['Age'] = df['Age'].fillna(
        df['Age'].mean()
      )
      df['Embarked'] = df['Embarked'].fillna(
        df['Embarked'].mode()[0]
      )
      
      df['Fare'] = df['Fare'].map(
        lambda i: np.log(i) if i > 0 else 0
      )

      logger.info('Applying one-hot encoding to categorical columns')
      categorical = ['Pclass', 'Sex', 'Embarked']

      for col in categorical:
        dummy = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummy], axis=1)
        df.drop(col, axis=1, inplace=True)
      
      logger.info('Dropping low variance columns')
      df.drop(
        labels=[
          'Embarked_Q',
          'SibSp',
          'Age',
          'Parch',
          'Pclass_2'
        ],
        axis='columns',inplace=True)
      transformed_df.append(df)

    return tuple(transformed_df)

  def scale(self, train_df, test_df):
    scaler = self.scaler.fit(train_df)
    logger.info('Scaling the data')
    return scaler.transform(train_df), \
      scaler.transform(test_df)

  def apply(self, train_df, test_df):
    train_df, test_df = self.transform(
      train_df, test_df
    )
    if self.scaler:
      train_df, test_df = self.scale(
        train_df, test_df
      )
    return train_df, test_df
