import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "type",
    help="aircraft type to find (e.g. C172)",
    type=str
)
args = parser.parse_args()

from opensky_api import OpenSkyApi
from traffic.data import aircraft
from dotenv import load_dotenv
from os import getenv
from time import time

# aircraft.download_opensky()


def get_env_var(var_name):
    var_value = getenv(var_name)
    if var_value is None:
        raise KeyError(f"{var_name} environment variable not found. do you have a .env file or have you exported the correct variables?")
    return var_value


load_dotenv()
username = get_env_var("OPENSKY_USERNAME")
password = get_env_var("OPENSKY_PASSWORD")

print("requesting opensky states")
# api = OpenSkyApi(username=username, password=password)
api = OpenSkyApi()
# print(api.get_states())
openskystates = api.get_states()
print("query complete")

found = []

start = time()
for state in openskystates.states:
    print(f"checking aircraft {state.callsign} ({state.icao24})")
    aircraft_info = aircraft.get(state.icao24)
    if aircraft_info is None:
        print("invalid aircraft")
        continue
    print(f"aircraft type {aircraft_info.model}")
    if aircraft_info.typecode == args.type:
        found.append(state.callsign)
        print("matching aircraft found")

print(f"state processing time: {time() - start} seconds")
# print(found)
print(len(found))
