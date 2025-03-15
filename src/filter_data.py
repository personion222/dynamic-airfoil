import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "type",
    help="aircraft type to look for (e.g. C172)",
    type=str
)
parser.add_argument(
    "data_dir",
    help="the directory of the csv datasets to check",
    type=str
)
parser.add_argument(
    "out_dir",
    help="the directory to output the filtered datasets",
    type=str
)
args = parser.parse_args()

from concurrent.futures import ThreadPoolExecutor
from traffic.data import aircraft
import pandas as pd
import time
import os


def fetch_model(icao):
    if icao in types:
        return
    if tail := aircraft.get(icao):
        types[icao] = tail.typecode


full_start = time.perf_counter()
print("collecting unique icao codes")
collect_start = time.perf_counter()
unique_icao = set()
for fp in args.data_dir.glob("*.csv"):
    df = pd.read_csv(fp, usecols=["icao24"], dtype={"icao24": "string"})
    unique_icao.update(df["icao24"].dropna().unique())
    print(f"collected icao codes from {fp.name}")
print(f"collected {len(unique_icao)} unique icao codes in {round(time.perf_counter() - collect_start)} sec")

print("getting typecodes from icao codes")
typecode_start = time.perf_counter()
types = {}

with ThreadPoolExecutor() as executor:
    promises = {executor.submit(fetch_model, icao): icao for icao in unique_icao}
    completed = 0
    for future in promises:
        future.result()
        completed += 1
        print(f" finished saving icao model pairs for {completed}/{len(unique_icao)} codes")
print(f"created {len(types)} icao model pairs in {round(time.perf_counter() - typecode_start)} sec")

print("filtering aircraft by model")
valid_start = time.perf_counter()
valid_icao = {icao for icao, t in types.items() if t == args.type}
print(f"filtered {len(valid_icao)} icao codes by model in {round(time.perf_counter() - valid_start)} sec")

print("filtering and writing csv files by valid icao codes")
filter_start = time.perf_counter()
total_files = len(os.listdir(args.data_dir))
processed_files = 0
for fp in os.listdir(args.data_dir):
    file_start = time.perf_counter()
    
    df = pd.read_csv(fp, dtype={"icao24": "string"}, low_memory=False, engine="c")
    
    filtered = df[df["icao24"].isin(valid_icao)]
    filtered.to_csv(args.out_dir / fp.name, index=False)
    
    print(f"wrote {fp.name} in {round(time.perf_counter() - file_start)} sec")
    processed_files += 1

print(f"wrote all files in {round(time.perf_counter() - filter_start)} sec")
print(f"completed processing in {round(time.perf_counter() - full_start)} sec")
