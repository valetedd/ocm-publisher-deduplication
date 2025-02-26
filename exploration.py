import rapidfuzz 
import polars as pl
import re
import matplotlib.pyplot as plt

# TODO: rappresentazione vettoriale + clustering (HDBSCAN) per individuare cluster di similarità da usare eventualmente nella deduplicazione


def get_clusters(): 
    pass

def main():
    processed_data = pl.read_parquet("data/processed_data.parquet")
    pub = processed_data.get_column("publisher").to_list()

if __name__ == "__main__":
    main()
