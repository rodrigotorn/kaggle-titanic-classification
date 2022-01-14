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
from src.models.knn import KNN
from src.models.rf import RF

# %% tags=[]
raw_train_df: pd.DataFrame = pd.read_csv('data/train.csv', index_col=0)
raw_test_df: pd.DataFrame = pd.read_csv('data/test.csv', index_col=0)
y_train: pd.Series = raw_train_df['Survived']

preprocesser = Preprocess(raw_train_df, StandardScaler())
x_train: np.ndarray = preprocesser.transform(raw_train_df)
x_test: np.ndarray = preprocesser.transform(raw_test_df)

mlp_params = [{'hidden_layer_sizes': [
    (15,), (20,), (30,), (40,), (50,),
  ]}]  
knn_params = [{'n_neighbors': [5, 10, 15, 20, 25]}]
rf_params = [{'n_estimators': [50, 100, 200, 500, 1000]}]
    
mlp = MLP(search_params=True, params=mlp_params)
knn = KNN(search_params=True, params=knn_params)
rf = RF(search_params=True, params=rf_params)

mlp_score = mlp.train(x_train, y_train)
knn_score = knn.train(x_train, y_train)
rf_score = rf.train(x_train, y_train)

print(mlp_score, knn_score, rf_score)

# %%
predictions: pd.DataFrame = pd.DataFrame({
    'mlp': mlp.predict(x_test),
    'knn': knn.predict(x_test),
    'rf': rf.predict(x_test),
})

predictions['PassengerId'] = raw_test_df.index
predictions['Survived'] = predictions.mode(axis=1)
predictions.drop(['mlp', 'knn', 'rf'], axis=1, inplace=True)
predictions.to_csv('data/output.csv', index=False)
