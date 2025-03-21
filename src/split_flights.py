import argparse

parser = argparse.ArgumentParser()
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

import pandas as pd
import numpy as np
import time
import os

# first, split aircraft flight data into chunks if the timestep is longer than
# MIN_SPLIT_TIMESTEP, and groundspeed is lower than MAX_SPLIT_GROUNDSPEED
MIN_SPLIT_TIMESTEP = 60 * 15 # sec
MAX_SPLIT_GROUNDSPEED = 25 # m/sec

# after, remove contiguous chunks if the timestep is longer than
# MIN_REMOVAL_TIMESTEP
MIN_REMOVAL_TIMESTEP = 60 # sec
MIN_LENGTH = 60 * 60 # sec

full_start = time.perf_counter()
for aircraft in os.listdir(args.data_dir):
    aircraft_start = time.perf_counter()

    df = pd.read_csv(args.data_dir + aircraft, sep=',')
    df.drop_duplicates(subset="lastposupdate")

    df["time_diff"] = df["lastposupdate"].diff()
    overlap_mask = (df["time_diff"] > MIN_SPLIT_TIMESTEP) & (df["velocity"] < MAX_SPLIT_GROUNDSPEED)

    split_indices = df[overlap_mask].index.tolist()

    chunks = [df.iloc[start: end] for start, end in zip([0] + split_indices, split_indices + [len(df)])]
    filtered_chunks = [chunk for chunk in chunks
                       if (chunk["time_diff"] < MIN_REMOVAL_TIMESTEP).any() and
                       chunk.loc[chunk.index[-1], "lastposupdate"] -
                       chunk.loc[chunk.index[0], "lastposupdate"] > MIN_LENGTH]

    for idx, chunk in enumerate(filtered_chunks):
        chunk.to_csv(f"{args.out_dir}{aircraft.split('.')[0]}_{idx}.csv")
    
    print(f"processed aircraft {aircraft.split('.')[0]} into {len(filtered_chunks)} chunk(s) ({len(chunks)} unfiltered) in {round((time.perf_counter() - aircraft_start) * 1000)} ms")

print(f"finished processing all aircraft in {round(time.perf_counter() - full_start)} sec")
