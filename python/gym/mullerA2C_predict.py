#!/usr/bin/env python3
# -*- coding: UTF8 -*-

from stable_baselines3 import A2C
from mullerenv import MullerEnv
import numpy as np

# import matplotlib.pyplot as plt

history = 3

model = A2C.load("sac_muller")
# plt.matshow(env.V)
env = MullerEnv(history=history)

print()

obs = env.reset()
traj = []
total_reward = 0
while True:
    action, _states = model.predict(obs, deterministic=False)
    # action, _states = model.predict(obs, deterministic=True)
    # print(obs)
    obs, reward, done, info = env.step(action)
    if env.V[env.discretized_coords] == env.V.min():
        done = True
    total_reward += reward
    # print(total_reward, reward, env.V[env.discretized_coords])
    traj.append(env.coords.copy())
    if done:
        obs = env.reset()
        break

traj = np.asarray(traj)
np.save('env.npy', env.V)
np.save('traj.npy', traj)
