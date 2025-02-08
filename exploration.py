import polars as pl
import io
import re 
from pprint import pprint as pp
import time
from datetime import datetime


def process_col(row : tuple):
    try:
        if not any(map(len, row)):
            return

        data_field = row[0]
        
        if "; " in data_field:
            process_col((data_field.split("; ")[0], "", ""))
            return
        
        pattern = re.compile(r"\[(?:\w+:[^\]]+)(?:\s+\w+:[^\]]+)*\]")
        id_part = pattern.search(data_field)
        id_vals = id_part.group() if id_part else ""


        if not id_vals:
            return
        
        lit = re.sub(pattern, repl="", string=data_field)
        

        if " " in id_vals:
            split = id_vals[1:].split(" ")
            omid, cr = split.pop(0), ", ".join(split).strip("]")
            # qualcosa di rotto qui
            # TODO: trovare metodo migliore per gestire casi in cui ci sono molteplici crossref
        else:
            omid, cr = id_vals[1:].strip(), ""

        return (lit.strip(), omid, cr)
    
    except Exception as e:
        print(f"Error occurred: {e}")
        time.sleep(5)




INPUT_PATH : str = "data/part-0.parquet"
raw = pl.read_parquet(INPUT_PATH).with_columns([
    pl.lit("").alias("pub_omid"),
    pl.lit("").alias("pub_cr")
])

print(raw.head())
print(f"{len(raw)} at {datetime.now().strftime('%H:%M:%S')}")

deduped_d = raw.unique(subset="publisher")
print(len(deduped_d))
data = deduped_d.map_rows(process_col)
data.columns = ["publisher", "pub_omid", "pub_cr"]
print(data.head(10))
print(f"Finished data processing at {datetime.now().strftime('%H:%M:%S')}\nGetting unique values for 'publisher' and 'pub_cr'")

final_df = data.unique(subset="publisher")
print(len(final_df))
with open("publishers.txt", mode="w", encoding="utf-8") as f:
    publishers = sorted(p+"\n" for p in final_df.get_column("publisher").to_list() if p is not None)
    f.writelines(publishers)