import polars as pl
import unicodedata as ucd
import re
from pprint import pprint as pp
import time
from datetime import datetime


def normalize(s: str):
    normalized_s = ucd.normalize("NFKC", s.strip().lower())
    return "".join([c for c in normalized_s if not ucd.combining(c)])


def process_row(row: tuple) -> tuple:

    try:
        if not any(map(len, row)):
            return

        data_field = row[0]
        
        if "; " in data_field:
            process_row((data_field.split("; ", maxsplit=1)[0], "", ""))
            return
        
        pattern = re.compile(r"\[(?:\w+:[^\]]+)(?:\s+\w+:[^\]]+)*\]")
        id_part = pattern.search(data_field)

        id_vals = id_part.group() if id_part else ""
        if not id_vals:
            return
        
        lit = re.sub(pattern, repl="", string=data_field)
        

        if " " in id_vals:
            split = id_vals[1:].split(" ")
            omid, cr = split.pop(0).strip("]"), ", ".join(split).strip("]")
        else:
            omid, cr = id_vals[1:].strip().strip("] "), ""

        return (normalize(lit), omid, cr)
    
    except Exception as e:
        print(f"Error occurred: {e}")
        time.sleep(5)


def main():

    INPUT_PATH : str = "data/part-0.parquet"

    raw = pl.read_parquet(INPUT_PATH).with_columns([
        pl.lit("").alias("pub_omid"),
        pl.lit("").alias("pub_cr")
    ])

    print(raw.head())
    print(f"{len(raw)} at {datetime.now().strftime('%H:%M:%S')}")

    deduped_d = raw.unique(subset="publisher")
    print(len(deduped_d))

    data = deduped_d.map_rows(process_row)
    data.columns = ["publisher", "pub_omid", "pub_cr"]
    print(data.head(10))
    print(f"Finished data processing at {datetime.now().strftime('%H:%M:%S')}\nGetting unique values for 'publisher'")

    final_df = data.unique(subset="publisher")
    print(len(final_df))

    with open("data/publishers.txt", mode="w", encoding="utf-8") as f:
        publishers = sorted(p+"\n" for p in final_df.get_column("publisher").to_list() if p is not None)
        f.writelines(publishers)

    final_df.write_parquet("data/processed_data.parquet")

if __name__ == "__main__":
    main()