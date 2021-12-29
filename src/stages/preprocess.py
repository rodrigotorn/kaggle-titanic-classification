from src.stage import Stage
import pandas as pd

class Preprocess(Stage):
  def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df[1,:]
    return df
