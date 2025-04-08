from geographiclib.geodesic import Geodesic
from metpy.units import units
import metpy.calc as mpcalc
from geopy import Point
from math import floor
import xarray as xr
import pandas as pd
import numpy as np
import rasterio
import bisect


def get_vars(var_arrays, distance_array, query_distance):
    out = []
    for var_array in var_arrays:
        # print(var_array)
        out.append(np.interp(query_distance, distance_array, var_array))
    return out


def find_coordinates(coord_array, distance_array, query_distance):
    match_idx = np.where(distance_array == query_distance)[0]
    insert_idx = bisect.bisect_left(distance_array, query_distance)
    if len(match_idx) > 0:
        lat, lon = coord_array[match_idx].tolist()[0]
        return lon, lat

    if insert_idx == 0:
        lat, lon = coord_array[0].tolist()
        return lon, lat

    if insert_idx == len(distance_array):
        lat, lon = coord_array[-1].tolist()
        return lon, lat

    left_idx = insert_idx - 1
    right_idx = insert_idx
    left_distance = distance_array[left_idx]
    left_coord = coord_array[left_idx]
    right_coord = coord_array[right_idx]

    geod = Geodesic.WGS84

    g = geod.Inverse(
        left_coord[0], left_coord[1],
        right_coord[0], right_coord[1]
    )

    bearing = g["azi1"]

    result = geod.Direct(
        left_coord[0], left_coord[1],
        bearing, query_distance - left_distance
    )

    return result["lon2"], result["lat2"]


class Topography:
    def __init__(self, raster_path):
        self.cache = {}
        self.raster_path = raster_path
        self.src = rasterio.open(raster_path)

    def sample_pixel(self, lon, lat):
        row, col = rasterio.transform.rowcol(self.src.transform, lon, lat)

        return self.src.read(1, window=(
            (row, row + 1),
            (col, col + 1)
        ))[0][0]

    def get_topography(self, flight_path: pd.DataFrame, step_dist: int):
        distances = flight_path["totaldist"].to_numpy()
        coordinates = flight_path[["lat", "lon"]].to_numpy()
        max_dist = distances.max()
        step_count = floor(max_dist / step_dist)
        topography_array = np.empty((step_count, 2), dtype=np.float32)

        for step in range(step_count):
            # print(find_coordinates(
            #     coordinates, distances, step * step_dist
            # ))
            topography_array[step] = np.array((step * step_dist, self.sample_pixel(*find_coordinates(
                coordinates, distances, step * step_dist
            ))))

        return topography_array

    def close(self):
        self.src.close()

class Weather:
    def __init__(self, netcdf_path):
        self.netcdf_path = netcdf_path

        # read dataset with xarray
        self.ds = xr.open_dataset(
            netcdf_path,
            engine="netcdf4",
            chunks={"valid_time": "auto", "pressure_level": "auto"},
            cache=True
        )

        # Initialize caches
        self.weather_cache = {}
        self.stack_cache = {}
        self.pressure_levels = self.ds["pressure_level"].values

        # print(self.ds["valid_time"].values)

        # Setup unit objects and precompute indices
        self._setup_units()
        self._precompute_indices()

    def _setup_units(self):
        '''create pint units for use with metpy'''
        self.meters_unit = units.meters
        self.pa_s_unit = units("Pa/s")
        self.hpa_unit = units.hPa
        self.kelvin_unit = units.degK
        self.kg_kg_unit = units("kg/kg")
        self.m2_s2_unit = units("m^2/s^2")

    def _precompute_indices(self):
        '''create index for pressure levels'''
        self.pressure_to_index = {p: i for i, p in enumerate(self.pressure_levels)}
        self.pressure_array = self.pressure_levels * self.hpa_unit

    def _get_weather_stack(self, lon, lat, sim_time, dt):
        '''Get and cache weather data for a location and time.'''
        stack_key = ((round(lon, 2), round(lat, 2)), round(sim_time / 1800) * 1800)

        if stack_key in self.stack_cache:
            return self.stack_cache[stack_key]

        weather_stack = self.ds.sel(
            valid_time=dt,
            latitude=lat,
            longitude=lon,
            method="nearest"
        ).compute()

        self.stack_cache[stack_key] = weather_stack
        if len(self.stack_cache) > 1024:
            self.stack_cache.pop(next(iter(self.stack_cache)))

        return weather_stack

    def _find_closest_pressure_level(self, geopotential_data, target_geopotential):
        '''Find the closest pressure level for a given geopotential height.'''
        geopot_with_units = geopotential_data * self.m2_s2_unit
        geopot_diff = np.abs(geopot_with_units - target_geopotential)
        return np.argmin(geopot_diff.magnitude)

    def get_weather(self, lon, lat, sim_time, geoalt):
        '''
        get the weather conditions at a specific point and time.

        parameters:
        lon : float - longitude (degrees)
        lat : float - latitude (degrees)
        sim_time : int - unix timestamp (seconds)
        geoalt : float - altitude above wgs84 ellipsoid (meters)

        returns dictionary of weather values
        '''

        # check cache for precalculated weather
        cache_key = (
            (round(lon, 2), round(lat, 2)),
            round(sim_time / 1800) * 1800,
            round(geoalt / 15) * 15
        )

        if cache_key in self.weather_cache:
            return self.weather_cache[cache_key]

        # select data from array if not in cache
        dt = pd.to_datetime(sim_time, unit='s', utc=True).tz_localize(None).to_datetime64()
        weather_stack = self._get_weather_stack(lon, lat, sim_time, dt)
        target_geopotential = mpcalc.height_to_geopotential(geoalt * self.meters_unit) # courtesy of metpy
        geopot_data = weather_stack['z'].values

        closest_idx = self._find_closest_pressure_level(geopot_data, target_geopotential)
        closest_pressure = self.pressure_levels[closest_idx]
        point_weather = weather_stack.sel(pressure_level=closest_pressure)

        # extract values and save to dictionary
        values = {
            'z': float(point_weather['z'].values.item()),
            'q': float(point_weather['q'].values.item()),
            'crwc': float(point_weather['crwc'].values.item()),
            'cswc': float(point_weather['cswc'].values.item()),
            't': float(point_weather['t'].values.item()),
            'u': float(point_weather['u'].values.item()),
            'v': float(point_weather['v'].values.item()),
            'w': float(point_weather['w'].values.item())
        }

        # convert vertical velocity with respect to pressure to that with respect to height
        mixing_ratio = mpcalc.mixing_ratio_from_specific_humidity(values['q'] * self.kg_kg_unit) # thanks metpy <3
        vertical_vel = float(mpcalc.vertical_velocity( # is there anything that metpy can't do
            values['w'] * self.pa_s_unit,
            closest_pressure * self.hpa_unit,
            values['t'] * self.kelvin_unit,
            mixing_ratio
        ).magnitude)

        # print(closest_pressure.item())
        # print(values['w'])

        # create output dictionary and save it in cache
        result = {
            'z': values['z'],
            'q': values['q'],
            'crwc': values['crwc'],
            'cswc': values['cswc'],
            't': values['t'],
            'u': values['u'],
            'v': values['v'],
            'w': vertical_vel,
            'p': closest_pressure
        }

        self.weather_cache[cache_key] = result

        return result

    def clear_caches(self):
        '''clear all caches to free memory'''
        self.weather_cache.clear()
        self.stack_cache.clear()

    def close(self):
        '''close the dataset and clear caches'''
        self.clear_caches()
        self.ds.close()


if __name__ == "__main__":
    topo_reader = Topography("../global_data/copernicus_glo90/COP90_hh.vrt")
    print(topo_reader.sample_pixel(49, 47.32))
    topo_reader.close()
    atm_reader = Weather("../global_data/copernicus_era5/e1e175ca5146ec705aab8b4973262cf3.nc")
    print(atm_reader.get_weather(170.1418, -43.5950, 1656319105, 4000))
    print(atm_reader.get_weather(170.1418, -43.5950, 1656317105, 8000))
    print(atm_reader.get_weather(170.1418, -45.5950, 1656317105, 4000))
    print(atm_reader.get_weather(170.1418, -43.5950, 1656317105, 4000))
    print(atm_reader.get_weather(170.1418, -43.5950, 1656317105, 4000))
    print(atm_reader.get_weather(170.1418, -43.5950, 1656317105, 4000))
    print(atm_reader.get_weather(170.1418, -43.5950, 1656317105, 4000))
