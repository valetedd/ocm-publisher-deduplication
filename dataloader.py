import tarfile
import polars as pl
import io
import os
from pathlib import Path
import pyarrow.dataset as ds
from functools import partial
from typing import List, Iterator
from datetime import datetime


def batched_members(tar_path: str, batch_size: int = 500) -> Iterator:

    with tarfile.open(tar_path, "r:gz") as tar:
        batch = []
        for member in tar:
            if member.name.startswith("csv_final/") and member.isfile():
                batch.append(member)
                if len(batch) >= batch_size:
                    yield batch 
                    batch = [] # resetting the batch after yield

        if batch: # Returning the final batch 
            yield batch


def extract_and_process(member: tarfile.TarInfo, tar: tarfile.TarFile, column_name: str) -> pl.DataFrame:

    try:

        file_obj = tar.extractfile(member)
        if file_obj is None:
            print(f"Skipping {member.name}, could not extract.")
            return None
        
        # Read into memory
        file_bytes = file_obj.read()
        df = pl.read_csv(io.BytesIO(file_bytes), 
                         columns=[column_name])

        return df
    
    except Exception as e:
        print(f"Error processing {member.name}: {e}")
        return None


def process_batch(tar_path: str, member_batch: List[tarfile.TarInfo], column_name: str, batch_id: int):

    results = []
    
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in member_batch:

            try:
                df = extract_and_process(member, tar, column_name)
                if df is not None:
                    results.append(df)

            except Exception as e:
                print(f"Error processing batch {batch_id}: {e}")

    if results:
        return pl.concat(results)
    

def combine_parquets(path: str | Path):

    if isinstance(path, str):
        path = Path(path)

    dataset = ds.dataset(path, format="parquet")
    ds.write_dataset(
        data=dataset,
        base_dir="data",
        format="parquet",
        create_dir=True
        )
    
    for f in path.iterdir():
        os.remove(f)

    try:
        os.rmdir(path)
    except OSError:
        print("Failed to delete parent directory of partitions")


def main() -> None:
    PATH = "meta_2024_06_20_csv.tar.gz"
    COLUMN_NAME = "publisher"
    BATCH_SIZE = 1000
    OUTPUT_DIR = "parquets"
    os.makedirs("parquets", exist_ok=True)

    print(f"Start at: {datetime.now().strftime('%H:%M:%S')}")

    process_func = partial(process_batch, tar_path=PATH, column_name=COLUMN_NAME)
    
        
    for batch_id, batch in enumerate(batched_members(PATH, batch_size=BATCH_SIZE)):
        df = process_func(member_batch=batch, batch_id=batch_id)
        print(f"Submitted batch {batch_id} at {datetime.now().strftime('%H:%M:%S')}.")
        df.write_parquet(f"{OUTPUT_DIR}/{batch_id}.parquet")

    print("Processing complete!")
    print("Concatenating parquets")

    combine_parquets(OUTPUT_DIR)


if __name__ == "__main__":
    main()
