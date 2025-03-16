# gets all the aircraft in the sky and saves them to a tinydb database
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "aircraft_type",
    help="the typecode of the aircraft to look for (e.g. C172)",
    type=str
)
parser.add_argument(
    "run_length",
    help="the length of time for which the api will be queried (HH:MM:SS)",
    type=str
)
parser.add_argument(
    "run_cooldown",
    help="how long the program should wait for before calling the api again (seconds)",
    type=int
)
parser.add_argument(
    "db_path",
    help="where to save the database (file path)",
    type=str
)
args = parser.parse_args()

from opensky_api import OpenSkyApi
from tinydb import TinyDB, where
from dotenv import load_dotenv
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

db = TinyDB(args.db_path)

api = OpenSkyApi(username=username, password=password)

start_time = time()
stop_flag = False
stop_thread = threading.Thread(target=stop_after_time, args=(time_to_seconds(args.run_length),))
stop_thread.start()
steps = 0

while not stop_flag:
    openskystates = api.get_states()
    for state in openskystates.states:
        if db.contains(where("icao") == state.icao24):
            old_states = db.get(where("icao") == state.icao24)["states"]
            db.update({
                "states": {
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
            }, where("icao") == state.icao24)
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
            db.insert({
                "icao": state.icao24,
                "callsign": state.callsign,
                "country": state.origin_country,
                "category": state.category,
                "states": aircraft_state
            })
    steps += 1
    pause.until(start_time + steps * args.run_cooldown)

stop_thread.join()
