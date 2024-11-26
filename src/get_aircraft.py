from opensky_api import OpenSkyApi
from dotenv import load_dotenv
import argparse
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
args = parser.parse_args()

api = OpenSkyApi(username=username, password=password)

s = api.get_states()
print(s.states[0])
