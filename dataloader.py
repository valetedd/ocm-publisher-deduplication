import os
import tarfile
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Iterator, List

import polars as pl
import pyarrow.dataset as ds


def batched_members(tar_stream: tarfile.TarFile, batch_size: int = 500) -> Iterator:

    # Yielding batches of TarInfo objects
    batch = []
    for member in tar_stream:
        if member.isfile():
            batch.append(member)
            if len(batch) >= batch_size: # checking if the batch size capacity was reached
                yield batch
                batch = []  # resetting the batch after yielding full batch

    if batch:  # Returning the final batch
        yield batch


def process_batch(
    tar_stream: tarfile.TarFile,
    member_batch: List[tarfile.TarInfo],
    batch_id: int,
):

    results = []

    for member in member_batch:

        try:
            file_obj = tar_stream.extractfile(member)
            if file_obj is None:
                print(f"Skipping {member.name}, could not extract.")
                return None

            # Read lazily
            lazy_df = pl.scan_csv(file_obj).select("publisher")

            if lazy_df.fetch(1).height > 0: # only fetches one row to verify the df is not empty
                results.append(lazy_df)

        except Exception as e:
            print(f"Error processing batch {batch_id}: {e}")

    if results:
        return pl.concat(results)


def combine_parquets(path: str | Path, output_dir):

    if isinstance(path, str):
        path = Path(path)
    elif not isinstance(path, Path):
        raise TypeError("'path' must either be a string path or a Path object")

    dataset = ds.dataset(path, format="parquet")
    ds.write_dataset(
        data=dataset, base_dir=output_dir, format="parquet", create_dir=True
    )

    for f in path.iterdir():
        os.remove(f)
    print("Removed partial parquets")
    try:
        os.rmdir(path)
    except OSError:
        print("Failed to delete parent directory of partitions")


def dump_to_parquet(
    path: str | Path,
    parquets_dir: str | Path = "./data/parquets/",
    output_dir: str | Path = "./data/merged_parquet/",
    batch_size: int = 500,
) -> str | Path:

    os.makedirs(parquets_dir, exist_ok=True)

    print(f"Start at: {datetime.now().strftime('%H:%M:%S')}")

    with tarfile.open(path, "r") as tar:

        process_func = partial(process_batch, tar_stream=tar)

        for batch_id, batch in enumerate(batched_members(tar, batch_size=batch_size)):
            df = process_func(member_batch=batch, batch_id=batch_id)
            if df is None:
                print("No data was yielded. Skipping iteration...")
                continue
            print(f"Processed batch {batch_id} at {datetime.now().strftime('%H:%M:%S')}.")
            df.sink_parquet(os.path.join(parquets_dir, f"{batch_id}.parquet"))

    print("Processing complete!")
    print("Concatenating parquets...")

    combine_parquets(parquets_dir, output_dir)
    return os.path.join(output_dir, "part-0.parquet")


if __name__ == "__main__":
    PATH = "meta_2025_02_13_csv.tar"
    COLUMN_NAME = "publisher"
    BATCH_SIZE = 1000
    dump_to_parquet(path=PATH, batch_size=BATCH_SIZE)
