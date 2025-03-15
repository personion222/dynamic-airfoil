import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "type",
    help="aircraft type to check for (e.g. C172)",
    type=str
)
parser.add_argument(
    "dataset",
    help="the directory of the csv dataset to check",
    type=str
)
# parser.add_argument(
#     "entries",
#     help="the amount of entries to read from the csv",
#     type=int
# )
args = parser.parse_args()

from traffic.data import aircraft
from time import perf_counter_ns
from itertools import repeat
import pandas as pd

print("loaded db")
icao_types = {}
row_count = 0
matching_aircraft = 0

with open(args.dataset, mode='r') as f:
    df = pd.read_csv(args.dataset)
    total = len(df.index)

    variables = list(df.columns)[2:-1] # remove query time, icao code, and lastcontact (always complete)
    general_completeness = dict(zip(variables, repeat(0)))
    aircraft_completeness = dict(zip(variables, repeat(0)))

    for row in df.to_dict(orient="records"):
        row_count += 1
        row_completeness = {variable: (1 if row[variable] else 0) for variable in variables}
        general_completeness = {variable: general_completeness[variable] + row_completeness[variable] for variable in variables}

        cache_start = perf_counter_ns()
        aircraft_type = icao_types.get(row["icao24"])
        if aircraft_type is None:
            new_start = perf_counter_ns()
            tail_obj = aircraft.get(row["icao24"])
            if tail_obj is None:
                # print(f"invalid aircraft: {row["callsign"]} ({row["icao24"]})")
                continue
            icao_types[row["icao24"]] = tail_obj.typecode
            aircraft_type = icao_types[row["icao24"]]
            # print(f"new:              {aircraft_type} in {round((perf_counter_ns() - new_start) / 1_000_000)} ms")
        else:
            pass
            # print(f"from cache:       {aircraft_type} in {round((perf_counter_ns() - cache_start) / 1_000_000)} ms")

        if aircraft_type == args.type:
            print(f"match found! {row_count} rows / {total} complete ({round(row_count / total * 100, 3)}%)")
            matching_aircraft += 1
            aircraft_completeness = {variable: aircraft_completeness[variable] + row_completeness[variable] for variable in variables}

print({variable: general_completeness[variable] / row_count * 100 for variable in variables})
print({variable: aircraft_completeness[variable] / matching_aircraft * 100 for variable in variables})
