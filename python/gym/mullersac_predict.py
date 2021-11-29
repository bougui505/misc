#!/usr/bin/env python3
# -*- coding: UTF8 -*-

from stable_baselines3 import SAC
from mullerenv import MullerEnv
import numpy as np
# import matplotlib.pyplot as plt

env = MullerEnv(maxiter=2000)
# plt.matshow(env.V)

model = SAC.load("sac_muller")

obs = env.reset()
env.coords = np.asarray([98., 27.])
traj = []
total_reward = 0
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if env.V[env.discretized_coords] == env.V.min():
        done = True
    total_reward += reward
    print(total_reward, reward, env.V[env.discretized_coords])
    traj.append(env.coords.copy())
    if done:
        obs = env.reset()
        env.coords = np.asarray([98., 27.])
        break
traj = np.asarray(traj)
np.save('env.npy', env.V)
np.save('traj.npy', traj)
