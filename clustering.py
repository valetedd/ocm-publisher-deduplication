import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import polars as pl
from sentence_transformers import SentenceTransformer
from torch.cuda import is_available as cuda_is_available

if cuda_is_available():
    DEVICE = "cuda"
    import cuml.cluster as hdbscan
    import cuml.manifold as umap
else:
    DEVICE = "cpu"
    import hdbscan
    from umap import umap_ as umap


def get_cluster_labels(
    emb: np.ndarray, metric="euclidean", cluster_selection_method="eom"
):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
    )
    labels = clusterer.fit_predict(emb)

    return labels


def get_embeddings(data: List[str], reduce: bool = True):

    print(f"Using device: {DEVICE}")

    model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

    print("Encoding data...")

    embeddings = model.encode(
        data, show_progress_bar=True, convert_to_numpy=True, device=DEVICE
    )
    if not reduce:
        return embeddings

    umap_reducer = umap.UMAP(n_components=20, metric="euclidean", random_state=42)
    embeddings = umap_reducer.fit_transform(embeddings)
    print("Reduced embeddings!")
    return embeddings


def is_cjk(text):
    # Unicode ranges for CJK characters
    cjk_ranges = [
        (0x4E00, 0x9FFF),  # CJK Unified Ideographs
        (0x3040, 0x309F),  # Hiragana
        (0x30A0, 0x30FF),  # Katakana
        (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
        (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
        (0xAC00, 0xD7AF),  # Hangul Syllables
        (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
        (0x2A700, 0x2B73F),  # CJK Unified Ideographs Extension C
        (0x2B740, 0x2B81F),  # CJK Unified Ideographs Extension D
        (0x2B820, 0x2CEAF),  # CJK Unified Ideographs Extension E
        (0x2CEB0, 0x2EBEF),  # CJK Unified Ideographs Extension F
        (0x30000, 0x3134F),  # CJK Unified Ideographs Extension G
        (0x31350, 0x323AF),  # CJK Unified Ideographs Extension H
        (
            0xFF00,
            0xFFEF,
        ),  # Half-width and Full-width Forms (includes Japanese punctuation)
    ]

    # Check if any character in text falls within CJK ranges
    for char in text:
        code = ord(char)
        if any(start <= code <= end for start, end in cjk_ranges):
            return True
    return False

def is_not_latin(text):
    """
    Checks if any character in the text falls within Cyrillic Unicode ranges.
    """
    # Unicode ranges for Cyrillic characters
    if is_cjk(text):
        return True

    cyrillic_ranges = [
        (0x0400, 0x04FF),  # Cyrillic
        (0x0500, 0x052F),  # Cyrillic Supplement
        (0x2DE0, 0x2DFF),  # Cyrillic Extended-A
        (0xA640, 0xA69F),  # Cyrillic Extended-B
        (0x1C80, 0x1C8F),  # Cyrillic Extended-C
    ]

    # Check if any character in text falls within Cyrillic ranges
    for char in text:
        code = ord(char)
        if any(start <= code <= end for start, end in cyrillic_ranges):
            return True
    return False


def cluster_data(
    input_data: pl.DataFrame | pd.DataFrame | str | Path,
    output_dir: str | Path = "./data/",
    emb_reduction: bool = True,
    clustering_metric="euclidean",
    clustering_method="eom",
):
    os.makedirs(output_dir, exist_ok=True)

    ### Generating clusters with custom functions

    if isinstance(input_data, (str, Path)):

        processed_data = pl.read_parquet(input_data)

    elif isinstance(input_data, (pl.DataFrame, pd.DataFrame)):

        processed_data = input_data

    else:
        raise TypeError("input_data can only by a path or a in-memory DataFrame object")

    raw = processed_data.get_column("publisher").to_list()
    filtered = [p for p in raw if p and not is_cjk(p)]  # removing CJK publishers
    emb = get_embeddings(
        filtered,
        reduce=emb_reduction,
    )
    labels = get_cluster_labels(
        emb, metric=clustering_metric, cluster_selection_method=clustering_method
    )

    print(type(labels))
    print(len(labels))

    ## Saving clusters

    labelled_data = pl.DataFrame({"publisher": filtered, "label": labels})
    labelled_data.write_csv(os.path.join(output_dir, "labels.csv"))

    return labelled_data


if __name__ == "__main__":

    INPUT_DIR = "./data/processed_data.parquet"
    OUTPUT_DIR = "./data/clusters/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    labelled_data = cluster_data(INPUT_DIR)

    # Iterating over every df grouped by their label
    # to save each similarity cluster in separate folders
    for label, df in labelled_data.group_by("label"):
        if label in {-1, "-1"}:
            continue
        df = df.sort("publisher")
        df.write_csv(os.path.join(OUTPUT_DIR, f"cluster{label}.csv"))

    print("Clusters saved!")
