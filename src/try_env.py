from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import PPO
from gym_env import AircraftEnv
import pygame


def make_aircraft_env():
    return AircraftEnv(True)


# env = DummyVecEnv([make_aircraft_env])
# model = PPO.load("../models/reshapeable_agent_4.zip")
# obs = env.reset()

# env.reset()

env = make_aircraft_env()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    keys = pygame.key.get_pressed()
    action = (
        int(keys[pygame.K_w]) - int(keys[pygame.K_s]),
        int(keys[pygame.K_RIGHT]) - int(keys[pygame.K_LEFT]),
        int(keys[pygame.K_DOWN]) - int(keys[pygame.K_UP])
    )
    # action, _states = model.predict(obs)
    obs, rew, term, trunc = env.step(action)
  
    # if term or trunc:
    #     env = AircraftEnv("../split_data/", 30, 2, (1280, 720))
    #     env.reset()
    #     print("reset")
    #     continue
