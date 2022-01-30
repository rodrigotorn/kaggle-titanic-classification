# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# This notebooks aims to explore the train dataset. The goal is to check for outliers, understand the distribution for each feature and apply transformations when necessary. At the end of this notebook we expect to obtain the template dataset after preprocessing.

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
raw: pd.DataFrame = pd.read_csv('data/train.csv', index_col=0)
print(f'Train dataset shape: {raw.shape}')
raw.head()

# %% [markdown]
# As a first iteration, we choose to drop the columns Name, Ticket and Cabin, since it would require a lot of preprocessing and we are not certain that these columns are required for a good prediction.

# %%
raw.drop(labels=['Name', 'Ticket', 'Cabin'], axis='columns', inplace=True)
raw.info()

# %% [markdown]
# The missing values for Age are going to be considered the mean. For Embarked, the mode.

# %%
raw['Age'] = raw['Age'].fillna(raw['Age'].mean())
raw['Embarked'] = raw['Embarked'].fillna(raw['Embarked'].mode()[0])

# %% [markdown]
# The histograms for each column are plotted bellow.

# %% tags=[]
fig, axs = plt.subplots(8, 1, figsize=(8, 20))
for col, ax in zip(raw.columns, axs):
    ax.hist(raw[col])
plt.show()

# %% [markdown]
# The boxplot indicates that the Fare column has a lot of outlier values. To correct this, the column is log tranformed.

# %%
raw.boxplot()

# %%
print(f"Fare skewness before log: {raw['Fare'].skew()}")
raw['Fare'] = raw['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
print(f"Fare skewness after log: {raw['Fare'].skew()}")

# %% [markdown]
# The categorical columns are transformed using one-hot encoding

# %%
sex_dummy = pd.get_dummies(raw['Sex'])
raw = pd.concat([raw, sex_dummy], axis=1)
raw.drop('Sex', axis=1, inplace=True)

# %%
embarked_dummy = pd.get_dummies(raw['Embarked'], prefix='Embarked')
raw = pd.concat([raw, embarked_dummy], axis=1)
raw.drop('Embarked', axis=1, inplace=True)

# %%
pclass_dummy = pd.get_dummies(raw['Pclass'], prefix='Pclass')
raw = pd.concat([raw, pclass_dummy], axis=1)
raw.drop('Pclass', axis=1, inplace=True)

# %%
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(raw.corr(), cmap='RdBu')

# %%
raw.corr()['Survived'].abs().sort_values()

# %% [markdown]
# Columns with low correlation are dropped, and we can see the template for the dataset after preprocessing.

# %%
raw.drop(labels=['Embarked_Q', 'SibSp', 'Age', 'Parch', 'Pclass_2'],
         axis='columns',inplace=True)
raw
