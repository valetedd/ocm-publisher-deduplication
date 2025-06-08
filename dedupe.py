import os

import pandas as pd
from polyfuzz import PolyFuzz

from clustering import cluster_data
from dataloader import dump_to_parquet
from preprocessing import process_data


def main():

    # Defining dir loactions where intermediate
    # and final data is going to be saved
    TAR_PATH = "./meta_2025_02_13_csv.tar"
    DATA_DIR = "./test/"
    RESULTS_DIR = "./results/"

    THRESHOLD = 0.75
    BATCH_SIZE = 1000  # number of compressed csvs to process at a time

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Preparing the dump to be loaded storing it in a parquet
    parquet_path = dump_to_parquet(path=TAR_PATH, batch_size=BATCH_SIZE)

    # Processing data to extract IDs, clean it and normalize it
    processed_parq_path = process_data(input_path=parquet_path, output_dir=DATA_DIR)

    # Applying ML pipeline to cluster the data with HDBSCAN
    clustered_data = cluster_data(processed_parq_path, output_dir=DATA_DIR)

    # Loppingo over data grouped by label
    relevant_matches = []
    for label, df in clustered_data.group_by("label"):

        # Skipping the -1 group (unclustered data)
        if label in {-1, "-1"}:
            continue

        pubs = df.get_column("publisher").to_list()
        if not pubs:
            continue

        try:

            # Using polyfuzz to measure string similraity
            model = PolyFuzz()  # defaults to n-gram based tf-idf
            model.match(pubs)
            matches = model.get_matches()  # XXX: Returns a Pandas DataFrame

            # Filtering out matches based on threshold and redundant comparations
            # Redundant comparations: A - A ; A - B and B - A (symmetric and reflexive comparisons)
            matches = matches[
                (matches["Similarity"] >= THRESHOLD) & (matches["From"] < matches["To"])
            ]
            if not matches.empty:
                relevant_matches.append(matches)

        except Exception as e:
            print(
                f"{e} caused by one of these strings: {"\n".join(pubs)}\n\nLogging them for further examinations in results directory"
            )
            # Logging problematic strings
            with open(
                "./results/problematicStrings.txt", mode="a", encoding="utf-8"
            ) as f:
                f.write("\n\n--- Begin Cluster ---\n")
                f.writelines(pubs)
                f.write("\n--- End Cluster ---\n")

    result = pd.concat(relevant_matches, sort=True)
    result.to_csv(os.path.join(RESULTS_DIR, "duplicates.csv"))


if __name__ == "__main__":
    main()
