# gets all the aircraft in the sky and saves them to a dictionary, then to a json at the end

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "run_length",
    help="the length of time for which the api will be queried (HH:MM:SS)",
    type=str
)
parser.add_argument(
    "run_cooldown",
    help="how long the program should wait for before calling the api again (seconds) (minimum of 10)",
    type=int
)
parser.add_argument(
    "db_path",
    help="where to save the database (file path)",
    type=str
)
args = parser.parse_args()

from datetime import datetime, timezone
from opensky_api import OpenSkyApi
from dotenv import load_dotenv
from json import dump
from time import time
from os import getenv
import threading
import pause

# aircraft.download_opensky()


def get_env_var(var_name):
    var_value = getenv(var_name)
    if var_value is None:
        raise KeyError(f"{var_name} environment variable not found. do you have a .env file or have you exported the correct variables?")
    return var_value

def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

def stop_after_time(seconds):
    global stop_flag
    pause.seconds(seconds)
    stop_flag = True


load_dotenv()
username = get_env_var("OPENSKY_USERNAME")
password = get_env_var("OPENSKY_PASSWORD")

db = dict()

api = OpenSkyApi(username=username, password=password)

start_time = time()
stop_flag = False
stop_thread = threading.Thread(target=stop_after_time, args=(time_to_seconds(args.run_length),))
stop_thread.start()
steps = 0

print(f"starting {datetime.now(timezone.utc).strftime("%A %Y-%m-%d %H:%M:%S")} utc")
print(f"estimated finish {datetime.fromtimestamp(timezone.utc, int(start_time) + time_to_seconds(args.run_length))} utc")

while not stop_flag:
    openskystates = api.get_states()
    for state in openskystates.states:
        if state.icao24 in db:
            old_states = db[state.icao24]["states"]
            db[state.icao24]["states"] = {
                "time": old_states["time"] + [state.time_position],
                "lon": old_states["lon"] + [state.longitude],
                "lat": old_states["lat"] + [state.latitude],
                "baro_alt": old_states["baro_alt"] + [state.baro_altitude],
                "on_ground": old_states["on_ground"] + [state.on_ground],
                "groundspeed": old_states["groundspeed"] + [state.velocity],
                "track": old_states["track"] + [state.true_track],
                "vert_rate": old_states["vert_rate"] + [state.vertical_rate],
                "geo_alt": old_states["geo_alt"] + [state.geo_altitude]
            }
        else:
            aircraft_state = {
                "time": [state.time_position],
                "lon": [state.longitude],
                "lat": [state.latitude],
                "baro_alt": [state.baro_altitude],
                "on_ground": [state.on_ground],
                "groundspeed": [state.velocity],
                "track": [state.true_track],
                "vert_rate": [state.vertical_rate],
                "geo_alt": [state.geo_altitude]
            }
            db.update({
                state.icao24: {
                    "callsign": state.callsign,
                    "country": state.origin_country,
                    "category": state.category,
                    "states": aircraft_state
                },
            })
    
    print(f"saved {len(openskystates.states)} states on {datetime.now().strftime("%A %Y-%m-%d %H:%M:%S")}")
    with open(args.db_path, 'w') as f:
        dump({
            "data_range": (round(start_time), round(time())),
            "aircraft": db
        }, f)
    steps += 1
    pause.until(start_time + steps * args.run_cooldown)

stop_thread.join()

print(f"started querying {datetime.now(timezone.utc).strftime("%A %Y-%m-%d %H:%M:%S")} utc")
print(f"finished saving data {datetime.now(timezone.utc).strftime("%A %Y-%m-%d %H:%M:%S")} utc")
print(f"estimated finish {datetime.fromtimestamp(timezone.utc, int(start_time) + time_to_seconds(args.run_length))} utc")
