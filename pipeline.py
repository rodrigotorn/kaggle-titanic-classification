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
from pandas import read_csv, DataFrame
from sklearn.preprocessing import StandardScaler
from src.preprocess import Preprocess
from src.models.mlp import MLP

# %% tags=[]
raw_train_df = read_csv('data/train.csv', index_col=0)
raw_test_df = read_csv('data/test.csv', index_col=0)
y_train = raw_train_df['Survived']

preprocesser = Preprocess(raw_train_df, StandardScaler())
X_train = preprocesser.transform(raw_train_df)
X_test = preprocesser.transform(raw_test_df)

mlp = MLP()
scores = mlp.train(X_train, y_train)
scores

# %%
y_pred = mlp.predict(X_test)

predictions: DataFrame = DataFrame()
predictions['PassengerId'] = raw_test_df.index
predictions['Survived'] = y_pred
predictions
