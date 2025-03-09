from opensky_api import OpenSkyApi
from dotenv import load_dotenv
from time import time, sleep
import argparse
import os


def get_env_var(var_name):
    var_value = os.getenv(var_name)
    if var_value is None:
        raise KeyError(f"{var_name} environment variable not found. do you have a .env file or have you exported the correct variables?")
    return var_value


load_dotenv()
username = get_env_var("OPENSKY_USERNAME")
password = get_env_var("OPENSKY_PASSWORD")

parser = argparse.ArgumentParser()
parser.add_argument(
    "run_length",
    help="the length of time for which the api will be queried (HH:MM:SS)",
    type=int
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

# info_table = """
#     CREATE TABLE aircraft_info(
#         icao CHAR(6) PRIMARY KEY,
#         callsign VARCHAR(15),
#         country VARCHAR(64),
#         aircraft_category INT
#     )
# """
# cur.execute(info_table)

api = OpenSkyApi(username=username, password=password)

s = api.get_states()
query_time = time()
state_info = [
    {
        "icao": state.icao24,
        "lat": state.latitude,
        "lon": state.longitude,
        "groundspeed": state.velocity,
        "callsign": state.callsign,
        "country": state.origin_country,
        "type": state.category,
        "time": query_time
    }
    for state in s.states
]

print('\n'.join(state_info))

# insert_aircraft = """
#     INSERT INTO aircraft_info (
#         icao,
#         callsign,
#         country,
#         aircraft_category
#     )
#     VALUES (?, ?, ?, ?);
# """
# cur.executemany(insert_aircraft, state_ls)

# con.commit()
