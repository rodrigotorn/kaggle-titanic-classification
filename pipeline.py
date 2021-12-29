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
from src.stages.ingest import Ingest
from src.stages.preprocess import Preprocess
from src.stages.model import Model

# %%
raw_train_df = Ingest().load('data/train.csv')
raw_test_df = Ingest().load('data/test.csv')

X_train, X_test = Preprocess().transform(raw_train_df, raw_test_df)
y_train = raw_train_df['Survived']

y_pred = Model().predict(X_train, y_train, X_test)

# %%
predictions: pd.DataFrame = pd.DataFrame()
predictions['PassengerId'] = raw_test_df.index
predictions['Survived'] = y_pred
predictions.to_csv('data/output.csv', index=False)
