import polars as pl
import os 
import statistics
from pprint import pprint as pp

INPUT_PATH : str = "data/part-0.parquet"
data = pl.read_parquet(INPUT_PATH).to_series()
pp(list(data.unique()))
