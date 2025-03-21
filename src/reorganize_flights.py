import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "data_dir",
    help="the directory of the csv datasets to look through",
    type=str
)
parser.add_argument(
    "out_dir",
    help="the directory to output the filtered datasets",
    type=str
)
args = parser.parse_args()

import pandas as pd
import time
import os

REQUIRED_COLUMNS = ["lat", "lon", "velocity", "heading", "vertrate", "onground", "baroaltitude", "geoaltitude"]

datasets = sorted(os.listdir(args.data_dir))

dfs = []
full_start = time.perf_counter()
for dataset in datasets:
    df_start = time.perf_counter_ns()
    dfs.append(pd.read_csv(args.data_dir + dataset, sep=','))
    print(f"read file {dataset} in {round((time.perf_counter_ns() - df_start) / 1_000_000)} ms")
read_end = time.perf_counter()

filter_write_start = time.perf_counter()
full_df = pd.concat(dfs, ignore_index=True)

full_df.dropna(subset=REQUIRED_COLUMNS, inplace=True)

filter_write_file_start = time.perf_counter_ns()
for icao, group in full_df.groupby("icao24"):
    unique_rows = group.drop_duplicates("lastposupdate", keep=False)
    if not unique_rows.empty:
        unique_rows.to_csv(f"{args.out_dir}{icao}.csv", index=False)
    print(f"filtered and wrote icao {icao} in {round((time.perf_counter_ns() - filter_write_file_start) / 1_000_000)} ms")
    filter_write_file_start = time.perf_counter_ns()

print(f"finished reading files in {round(read_end - full_start)} sec, filtering in {round(time.perf_counter() - filter_write_start)} sec")
print(f"finished processing in {round(time.perf_counter() - full_start)} sec")
