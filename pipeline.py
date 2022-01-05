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
from sklearn.preprocessing import StandardScaler
from src.stages.preprocess import Preprocess
from src.stages.mlp_predict import MLPPredict

# %% tags=[]
raw_train_df = pd.read_csv('data/train.csv', index_col=0)
raw_test_df = pd.read_csv('data/test.csv', index_col=0)
y_train = raw_train_df['Survived']

preprocesser = Preprocess(raw_train_df, StandardScaler())
X_train = preprocesser.transform(raw_train_df)
X_test = preprocesser.transform(raw_test_df)

y_pred = MLPPredict().predict(X_train, y_train, X_test)

# %%
predictions: pd.DataFrame = pd.DataFrame()
predictions['PassengerId'] = raw_test_df.index
predictions['Survived'] = y_pred
predictions
