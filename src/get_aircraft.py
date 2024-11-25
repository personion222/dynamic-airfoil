from opensky_api import OpenSkyApi
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "run_interval",
    help="how often the API should be queried (seconds)",
    type=int
)
parser.add_argument(
    "run_time",
    help="how long the program should run for (seconds)",
    type=int
)
args = parser.parse_args()

print(args.run_interval)

api = OpenSkyApi()

# s = api.get_states()
# print(s.states[0])
