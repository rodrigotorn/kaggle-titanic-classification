from src.base_stage import BaseStage
import pandas as pd

class Ingest(BaseStage):
  
  def transform(self, df: pd.DataFrame) -> pd.DataFrame:
    pass

  def load(self, path: str) -> pd.DataFrame:
    df  = pd.read_csv(path, index_col=0)
    return df
