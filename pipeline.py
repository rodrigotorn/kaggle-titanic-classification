# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 1

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.preprocess import Preprocess
from src.models.mlp import MLP

# %% tags=[]
raw_train_df: pd.DataFrame = pd.read_csv('data/train.csv', index_col=0)
raw_test_df: pd.DataFrame = pd.read_csv('data/test.csv', index_col=0)
y_train: pd.Series = raw_train_df['Survived']

preprocesser = Preprocess(raw_train_df, StandardScaler())
x_train: np.ndarray = preprocesser.transform(raw_train_df)
x_test: np.ndarray = preprocesser.transform(raw_test_df)

mlp = MLP()
scores: np.ndarray = mlp.train(x_train, y_train)
scores

# %%
y_pred: np.ndarray = mlp.predict(x_test)

predictions: pd.DataFrame = pd.DataFrame()
predictions['PassengerId'] = raw_test_df.index
predictions['Survived'] = y_pred
predictions
