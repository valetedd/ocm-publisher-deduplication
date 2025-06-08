import os
import re
import string
import time
import unicodedata as ucd
from datetime import datetime
from pathlib import Path

import polars as pl
from _typeshed import StrOrBytesPath, StrPath


def normalize(s: str):
    normalized_s = ucd.normalize("NFKD", s)
    return "".join(
        [c for c in normalized_s if not ucd.combining(c) or c in string.punctuation]
    )


def process_row(row: tuple) -> tuple | None:
    """
    Function used in conjunction with 'map_rows' to process the aggregated publisher data.

    Each row is assumed to be a tuple of shape '(index, data, emptyOmidID, emptyCrossRefID)'.
    The ID fields are to be extracted from the data with a regex.

    Returns:
    - A tuple of shape '(index, literal, OmidID, CrossRefID)' if the regex was successfully applied
    - A tuple of same shape but with 'index' in placec of one of the IDs if the regex did not match some id pattern
    """
    ### UNCOMMENT TO print each row ROWS
    # print(row)

    try:
        if not row[1]:
            return

        # index used to fill up empty entries, allowing deduplication
        # further down the pipelline
        idx = str(row[0])
        data_field = row[1].strip().lower()

        # Handling entries which present duplicated data (separated by ;)
        if "; " in data_field:
            data_field = data_field.split("; ", maxsplit=1)[0]

        # Regex pattern to match id values in the data field
        # strings of format [omid:ID] OR [omid:ID crossref:ID2]
        # NOTE: crossref can appear multiple times
        pattern = re.compile(r"\[(?:\w+:[^\]]+)(?:\s+\w+:[^\]]+)*\]")

        id_substr = pattern.search(data_field)  # Return Match object
        id_vals = id_substr.group() if id_substr else None  # stringifying the Match

        # Handling the case of missing IDs (both)
        if not id_vals:
            return (
                idx,
                data_field,
                idx,
                idx,
            )  # using index values to fill the ID fields

        # getting the literal component by filtering a substring using the id pattern
        lit = (
            re.sub(pattern, repl="", string=data_field).strip().strip("[").strip("]")
        )  # whitespaces and additional brackets are removed

        # Checking whether both Omid and CrossRef IDs were matched or just Omid
        # NOTE: If some IDs were found using the regex, Omid must be present but not CR
        if " " in id_vals:
            # Omid and CrossRef IDs are space-separated in the data field
            split = id_vals[1:].split(
                " "
            )  # starting from idx 1 to eliminate whitespace/bracket

            # Sometimes, there are two brackets at the start/end of the id_vals, so we try again to
            # strip them
            omid = split.pop(0).strip("[")
            cr = "; ".join(split).strip("]")
        else:
            omid = id_vals.strip("[").strip("]")
            cr = idx

        # Normalizing strings
        return (idx, normalize(lit), omid, cr)

    except Exception as e:
        print(f"Error occurred: {e}")
        time.sleep(5)


def process_data(input_path, output_dir: str | Path = "./data"):

    # Adding two empty columns to DataFrame for IDs.
    # IDS are going to be extracted from the only
    # column of the loaded file ('publisher'), which
    # aggregates all data in a specific format
    raw = (
        pl.scan_parquet(input_path)
        .with_columns([pl.lit("").alias("pub_omid"), pl.lit("").alias("pub_cr")])
        .with_row_index(
            "row_id"
        )  # adding a row index to help in processing (see process_row)
        .unique(
            subset="publisher"
        )  # eliminating all duplicate aggregated entries in advance
        .collect()
    )
    print(len(raw))

    print(raw.head())
    print(f"{len(raw)} at {datetime.now().strftime('%H:%M:%S')}")

    # Processing all data by row
    processed = raw.map_rows(process_row)
    processed.columns = ["row_id", "publisher", "pub_omid", "pub_cr"]
    print(processed.head(10))
    print(
        f"Finished data processing at {datetime.now().strftime('%H:%M:%S')}\nGetting unique values for 'publisher'"
    )

    # Getting unique values from each column
    # IDs itself are not sufficient because different literals of one publisher may be tied to different IDs
    deduped_df = (
        processed.unique(subset="pub_cr")
        .unique(subset="pub_omid")
        .unique(subset="publisher")
    )
    print(len(deduped_df))

    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, "publishers.txt"), mode="w", encoding="utf-8"
    ) as f:
        publishers = sorted(
            p + "\n"
            for p in deduped_df.get_column("publisher").to_list()
            if p is not None
        )
        f.writelines(publishers)

    output_filepath = os.path.join(output_dir, "processed_data.parquet")
    deduped_df.write_parquet(output_filepath)
    deduped_df.write_csv(os.path.join(output_dir, "processed_data.csv"))
    return output_filepath


if __name__ == "__main__":
    INPUT_PATH: str = "./merged_parquet/part-0.parquet"
    process_data(input_path=INPUT_PATH)
