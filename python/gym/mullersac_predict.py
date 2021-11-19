#!/usr/bin/env python3
# -*- coding: UTF8 -*-

from stable_baselines3 import SAC
from mullerenv import MullerEnv
import numpy as np
# import matplotlib.pyplot as plt

env = MullerEnv(maxiter=400)
# plt.matshow(env.V)

model = SAC.load("sac_muller")

obs = env.reset()
env.coords = np.asarray([27., 98.])
traj = []
total_reward = 0
while True:
    action, _states = model.predict(obs, deterministic=False)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    print(reward)
    traj.append(env.coords.copy())
    if done:
        obs = env.reset()
        env.coords = np.asarray([27., 98.])
        break
traj = np.asarray(traj)
np.save('env.npy', env.V)
np.save('traj.npy', traj)
