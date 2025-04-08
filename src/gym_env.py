from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes, ProgressBarCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3 import PPO
from xfoil import XFoil, model
from metpy.units import units
from time import perf_counter
import metpy.calc as mpcalc
from airfoil_funcs import *
from read_geodata import *
from random import shuffle
import pymunk.pygame_util
from meteo_funcs import *
import gymnasium as gym
import pandas as pd
import numpy as np
import pymunk
import pygame
import math
import os


class AircraftEnv(gym.Env):
    def __init__(self, render_on=False):
        self.dt = 1 / 15
        self.ppm = 1
        self.flight_dir = "../split_data/"
        self.flight_csvs = [self.flight_dir + f for f in os.listdir(self.flight_dir)]
        shuffle(self.flight_csvs)
        self.flight_idx = 0
        self.render_rez = (1280, 720)
        self.render_on = render_on

        self.topo_reader = Topography("../global_data/copernicus_glo90/COP90_hh.vrt")
        self.atm_reader = Weather("../global_data/copernicus_era5/e1e175ca5146ec705aab8b4973262cf3.nc")

        self.observation_space = gym.spaces.Dict({
                "groundvel": gym.spaces.Box(low=-85, high=85, shape=(2,), dtype=float),
                "airvel": gym.spaces.Box(low=-85, high=85, shape=(2,), dtype=float),
                "angular_velocity": gym.spaces.Box(low=-math.pi, high=math.pi, shape=(1,), dtype=float),
                "pitch": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=float),
                "aoa": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=float),
                "density": gym.spaces.Box(low=0.5, high=1.5, shape=(1,), dtype=float),
                "agl": gym.spaces.Box(low=0, high=4500, shape=(1,), dtype=float),
                "target_alt_dist": gym.spaces.Box(low=-4500, high=4500, shape=(1,), dtype=float)
        })

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=float)

        self.xf_wing = XFoil()
        self.xf_tail = XFoil()
        self.x2412, self.y2412 = scale_airfoil(*read_airfoil("../resources/2412.svg"))
        self.x0012up, self.y0012up = scale_airfoil(*read_airfoil("../resources/0012-up.svg"))
        self.x0012mid, self.y0012mid = scale_airfoil(*read_airfoil("../resources/0012.svg"))
        self.x0012down, self.y0012down = scale_airfoil(*read_airfoil("../resources/0012-down.svg"))
        self.xf_wing.airfoil = model.Airfoil(self.x2412, self.y2412)
        self.xf_tail.airfoil = model.Airfoil(self.x0012mid, self.y0012mid)
        # self.xf_tail.naca("0012")
        # self.xf_wing.naca("2412")
        self.xf_wing.max_iter = 100
        self.xf_wing.print = False
        self.xf_tail.print = False

        pygame.init()
        self.space = pymunk.Space()
        self.space.gravity = 0, -9.81 # gravity in meters/sec^2
        if self.render_on:
            self.screen = pygame.display.set_mode(self.render_rez)
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.scaling_transform = pymunk.Transform.scaling(self.ppm)
        # print(ppm)

    def set_collision(self, _, __, ___):
        self.colliding = True
        return True

    def reset_collision(self, _, __, ___):
        self.colliding = False

    def apply_passive_forces(self, obs, weather, heading):
        self.xf_wing.M = self.aircraft_body.velocity.length / (331 + 0.61 * (weather['t'] * units.degK).to("degC").magnitude)
        self.xf_wing.Re = math.hypot(*obs["airvel"]) * 1.472 / kinematic_visc(
            weather['t'] * units.degK,
            weather['p'] * units.hPa,
            weather['q'] * units("kg/kg")
        )
        cl, cd, cm, cp, conv =  self.xf_wing.a(-math.degrees(math.asin(obs["aoa"][0])))
        tail_cl, tail_cd, _, __, ___ = self.xf_tail.a(-math.degrees(math.asin(obs["aoa"][0])))

        # print("aoa    ", -math.degrees(math.asin(obs["aoa"][0])))

        if np.isnan(cl) or np.isnan(cd) or np.isnan(tail_cl) or np.isnan(tail_cd):
            return False

        # square_vel = self.aircraft_body.velocity.get_length_sqrd()

        # cl = max(-1, min(1, cl))
        # cd = max(-1, min(1, cd))
        # print("mach    ", self.xf_wing.M)
        # print("Re      ", self.xf_wing.Re)
        # print("cl      ", cl)
        # print("cd      ", cd)
        # print("density ", obs["density"])

        direct_horizontal_windspeed = pymunk.Vec2d(weather["u"], weather["v"]).dot(heading)
        wind = pymunk.Vec2d(direct_horizontal_windspeed, weather['w'])
        # print("wind    ", wind)

        wind_relative_vel = self.aircraft_body.velocity - wind

        square_vel = wind_relative_vel.get_length_sqrd()
        # print("windvel2", square_vel)

        if not (-0.866 < obs["aoa"][0] < 0.866):
            return False
        if wind_relative_vel.x < 0:
            return False
        if square_vel > 6619:
            return False
        
        q = 0.5 * obs["density"] * square_vel
        self.aircraft_body.torque = cm * q * 16.17 * 1.472
        # print("torque  ", self.aircraft_body.torque)

        lift_magnitude = cl * obs["density"] * square_vel * 16.17 * 0.5
        drag_magnitude = cd * obs["density"] * square_vel * 16.17 * 0.5
        tail_lift_magnitude = tail_cl * obs["density"] * square_vel * 2 * 0.5
        tail_drag_magnitude = tail_cd * obs["density"] * square_vel * 2 * 0.5

        # print("lift    ", lift_magnitude)
        # print("drag    ", drag_magnitude)

        # print("vel wind", wind_relative_vel)

        # lift_direction = wind_relative_vel.perpendicular_normal()
        # lift_force = lift_direction * lift_magnitude
        # drag_direction = -wind_relative_vel.normalized()
        # drag_force = drag_direction * drag_magnitude

        # net_force = lift_force + drag_force
        # print("force   ", net_force)
        self.aircraft_body.apply_force_at_local_point(pymunk.Vec2d(-drag_magnitude, lift_magnitude), (0, 0))
        self.aircraft_body.apply_force_at_local_point(pymunk.Vec2d(-tail_drag_magnitude, tail_lift_magnitude), (-5, 0))

        # if not conv:
        #     print(cl, cd, cm, cp)

        return True

    def _get_obs(self):
        # print("position", self.aircraft_body.position)
        # print("velocity", self.aircraft_body.velocity)
        if np.isnan(self.aircraft_body.position.x):
            lon, lat = find_coordinates(self.coordinates, self.distances, self.last_x)
        else:
            self.last_x = self.aircraft_body.position.x
            lon, lat = find_coordinates(self.coordinates, self.distances, self.last_x)
        lon, lat = find_coordinates(self.coordinates, self.distances, self.aircraft_body.position.x)
        sin_heading, cos_heading, target_alt = get_vars((
            np.sin(self.headings),
            np.cos(self.headings),
            self.altitudes
        ), self.distances, self.aircraft_body.position.x)
        topo_height = self.topo_reader.sample_pixel(lon, lat)
        weather_dict = self.atm_reader.get_weather(
            lon, lat,
            self.seconds,
            self.aircraft_body.position.y
        )
        # print(weather_dict)
        unit_heading = pymunk.Vec2d(cos_heading, sin_heading)
        direct_horizontal_windspeed = pymunk.Vec2d(weather_dict["u"], weather_dict["v"]).dot(unit_heading)

        wind = pymunk.Vec2d(direct_horizontal_windspeed, weather_dict['w'])
        vel = self.aircraft_body.velocity
        pitch = self.aircraft_body.angle
        wind_relative_vel = vel - wind
        relative_angle = math.atan2(wind_relative_vel.y, wind_relative_vel.x)
        aoa = relative_angle - self.aircraft_body.angle

        return weather_dict, unit_heading, {
            "groundvel": tuple(vel),
            "airvel": tuple(wind_relative_vel),
            "angular_velocity": self.aircraft_body.angular_velocity,
            "pitch": (math.sin(pitch), math.cos(pitch)),
            "aoa": (math.sin(aoa), math.cos(aoa)),
            "density": mpcalc.density(
                weather_dict['p'] * units.hPa,
                weather_dict['t'] * units.degK,
                mpcalc.mixing_ratio_from_specific_humidity(weather_dict['q'] * units("kg/kg"))
            ).magnitude,
            "agl": self.aircraft_body.position.y - topo_height,
            "target_alt_dist": target_alt - self.aircraft_body.position.y
        }

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed)
        # print("reset!!!")

        try:
            self.aircraft_body
            try:
                self.space.remove(self.aircraft_body, self.aircraft_collider)
            except AssertionError:
                pass
                # print("aircraft removal not required")
            try:
                for shape in list(self.space.static_body.shapes):
                    self.space.remove(shape)
            except AssertionError:
                pass
                # print("ground removal not required")
        except AttributeError:
            pass
            # print("deletion not required")

        self.seed = seed
        self.options = options
        self.flight_path = pd.read_csv(self.flight_csvs[self.flight_idx])
        self.flight_idx += 1
        self.flight_idx %= len(self.flight_csvs)
        self.distances = self.flight_path["totaldist"].to_numpy()
        self.coordinates = self.flight_path[["lat", "lon"]].to_numpy()
        self.headings = np.deg2rad(self.flight_path["heading"].to_numpy() - 90)
        self.altitudes = self.flight_path["geoaltitude"].to_numpy()

        if self.topo_reader.sample_pixel(self.flight_path.loc[0]["lon"], self.flight_path.loc[0]["lat"]) > self.flight_path.loc[0]["geoaltitude"]:
            return self.reset(seed, options)

        self.seconds = self.flight_path.loc[0]["time"]
        self.ticks = 0

        self.topography = self.topo_reader.get_topography(self.flight_path, 180)

        # print(self.topography)

        self.ground = self.space.static_body
        for idx, point in enumerate(self.topography[: -1]):
            if idx == 0:
                segment = pymunk.Segment(self.ground, (tuple(point)[0] - 100, tuple(point)[1]), tuple(self.topography[idx + 1]), 4)
            else:
                segment = pymunk.Segment(self.ground, tuple(point), tuple(self.topography[idx + 1]), 4)
            segment.friction = 0.6
            segment.elasticity = 0.3
            self.space.add(segment)
            segment.color = pygame.Color("green")

        self.aircraft_body = pymunk.Body()
        self.aircraft_body.mass = 754.3 # mass in kg
        self.aircraft_body.moment = 2427.3 # lateral moment of intertia in kg*m^2
        self.aircraft_collider = pymunk.Poly(self.aircraft_body, (
            (2.46, -0.0),
            (1.84, -0.48),
            (1.38, -0.68),
            (0.36, -0.68),
            (-4.82, 0.12),
            (-5.38, 0.27),
            (-5.84, 1.93),
            (-5.12, 1.93),
            (0.21, 0.96)
        ))
        self.aircraft_body.position = self.flight_path.loc[0]["totaldist"], self.flight_path.loc[0]["geoaltitude"]
        self.aircraft_body.velocity = pymunk.Vec2d(self.flight_path.loc[0]["velocity"], 0)
        self.aircraft_collider.friction = 0.6
        self.aircraft_collider.elasticity = 0.3
        self.aircraft_collider.color = pygame.Color("blue")
        self.colliding = False

        self.handler = self.space.add_wildcard_collision_handler(self.aircraft_collider.collision_type)
        self.handler.begin = self.set_collision
        self.handler.separate = self.reset_collision

        self.space.add(self.aircraft_body, self.aircraft_collider)

        _, __, observation = self._get_obs()

        return observation, {}


    def step(self, action):
        # print(action)
        reward = 0
        terminated = False
        truncated = False
        if self.colliding:
            terminated = True
        self.xf_wing.airfoil = model.Airfoil(self.x2412, self.y2412 * (action[1] / 2 + 1))

        if action[2] > 0:
            self.xf_tail.airfoil = model.Airfoil(*interp_paths(self.x0012mid, self.y0012mid, self.x0012up, self.y0012up, action[2]))
        else:
            self.xf_tail.airfoil = model.Airfoil(*interp_paths(self.x0012mid, self.y0012mid, self.x0012down, self.y0012down, -action[2]))

        weather, heading, observation = self._get_obs()

        ret = self.apply_passive_forces(observation, weather, heading)

        if not ret:
            terminated = True
            reward -= 100

        if self.ticks > 1800:
            truncated = True

        self.aircraft_body.apply_force_at_local_point(pymunk.Vec2d(2835.7 * (action[0] / 2 + 0.5), 0), (0, 0))
        
        if self.render_on:
            self.screen.fill((0, 0, 0))
            self.translation_transform = pymunk.Transform.translation(-self.aircraft_body.position[0] + 60, -self.aircraft_body.position[1] + 600)
            self.draw_options.transform = self.scaling_transform @ self.translation_transform
            self.space.debug_draw(self.draw_options)
            self.screen.blit(pygame.transform.flip(self.screen, False, True), (0, 0))
            pygame.display.update()

        self.space.step(self.dt)
        self.seconds += self.dt
        self.ticks += 1

        reward -= observation["target_alt_dist"] ** 2 / 10_000_000
        reward -= action[0] ** 2 * 0.015
        reward += self.aircraft_body.position.x * 0.001
        reward += observation["agl"] * 0.0001
        reward += self.ticks * 0.02

        if np.isnan(reward):
            reward = -150

        if terminated:
            print(f"terminated :( survived for {self.ticks} ticks, made it {self.aircraft_body.position.x} meters")
        
        if truncated:
            print(f"timed out, made it {self.aircraft_body.position.x} meters")

        return observation, reward, terminated, truncated, observation

def make_env():
    env = AircraftEnv()
    # env = gym.wrappers.NormalizeReward(env)
    return env


if __name__ == "__main__":
    env = make_vec_env(make_env, n_envs=16)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    env = AircraftEnv()
    # env.reset()
    # running = True
    # while running:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False

    #     out = env.step(env.action_space.sample())
        
    #     if not out:
    #         env = AircraftEnv("../split_data/", 30, 2, (1280, 720))
    #         env.reset()
    #         print("reset")
    #         continue

    checkpoint_callback = CheckpointCallback(
        save_freq=16,
        save_path="../models/temp_logs",
        name_prefix="dynamic_log",
        save_replay_buffer=True,
        save_vecnormalize=True
    )
    # stop_callback = StopTrainingOnMaxEpisodes(max_episodes=128)
    # progress_callback = ProgressBarCallback()
    # callback = CallbackList([checkpoint_callback, stop_callback, progress_callback])

    training_logger = configure("../logs/", ["stdout", "csv", "tensorboard"])

    rl_model = PPO("MultiInputPolicy", env, n_steps=4096, verbose=1)
    rl_model.set_parameters("../models/reshapeable_agent_4.zip")
    rl_model.set_logger(training_logger)
    rl_model.learn(total_timesteps=100_000, callback=checkpoint_callback, progress_bar=True)
    rl_model.save("../models/reshapeable_agent_5")
