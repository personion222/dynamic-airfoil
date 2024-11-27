from opensky_api import OpenSkyApi
from dotenv import load_dotenv
import argparse
import sqlite3
import os


def get_env_var(var_name, var_type):
    var_value = os.getenv(var_name)
    if var_value is None:
        raise KeyError(f"{var_type} environment variable not found. do you have a .env file or have you exported the correct variables?")
    return var_value


load_dotenv()
username = get_env_var("OPENSKY_USERNAME", "username")
password = get_env_var("OPENSKY_PASSWORD", "password")

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
parser.add_argument(
    "db_path",
    help="where to save the sqlite database (file path)",
    type=str
)
args = parser.parse_args()

con = sqlite3.connect(args.db_path)
cur = con.cursor()

info_table = """
    CREATE TABLE aircraft_info(
        icao CHAR(6),
        callsign VARCHAR(15),
        country VARCHAR(64),
        aircraft_category INT
    )
"""
cur.execute(info_table)

api = OpenSkyApi(username=username, password=password)

s = api.get_states()
state_ls = [
    (
        state.icao24,
        state.callsign,
        state.origin_country,
        state.category
    )
    for state in s.states
]

insert_aircraft = """
    INSERT INTO aircraft_info (
        icao,
        callsign,
        country,
        aircraft_category
    )
    VALUES (?, ?, ?, ?);
"""
cur.executemany(insert_aircraft, state_ls)

con.commit()
