from src.base_stage import BaseStage
import pandas as pd

class Preprocess(BaseStage):
  def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df[1,:]
    return df
