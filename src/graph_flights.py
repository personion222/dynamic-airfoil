import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "data_dir",
    help="the directory of the csv datasets to check",
    type=str
)
args = parser.parse_args()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import os

fig, ax1 = plt.subplots()

ax1.set_xlabel("distance travelled (m)")
ax1.set_ylabel("altitude (m)")

ax2 = ax1.twinx()

ax2.set_ylabel("groundspeed (m/s)")

lines = []

for flight_name in random.sample(os.listdir(args.data_dir), 100):
    flight = pd.read_csv(args.data_dir + flight_name)
    print(flight_name)

    ax1.plot(
        flight["totaldist"], flight["geoaltitude"],
        label=f"altitude {flight_name.split('.')[0]}",
        linestyle="solid")
    # ax.plot(((flight["lastcontact"] - flight["lastcontact"].min()) / (flight["lastcontact"].max() - flight["lastcontact"].min())) * flight["totaldist"].max(), flight["geoaltitude"], label=f"geoaltitude {flight_name.split('.')[0]} vs time")
    ax2.plot(
        flight["totaldist"], flight["velocity"],
        label=f"groundspeed {flight_name.split('.')[0]}",
        linestyle="dashed")

ax1.set_ylim(bottom=0)
ax2.set_ylim(bottom=0)
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
ax1.ticklabel_format(useOffset=False, style="plain")
ax2.ticklabel_format(useOffset=False, style="plain")
plt.show()
