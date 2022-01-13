import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from test.preprocess_fixtures import expected_preprocessed_sample
from src.preprocess import Preprocess

def test_preprocess(expected_preprocessed_sample):
  raw_train_df = pd.read_csv('data/train.csv', index_col=0)
  preprocesser = Preprocess(raw_train_df, StandardScaler())
  actual_preprocessed_sample = pd.DataFrame(
    preprocesser.transform(raw_train_df)[0:4])

  pd.testing.assert_frame_equal(
    actual_preprocessed_sample,
    expected_preprocessed_sample
  )
