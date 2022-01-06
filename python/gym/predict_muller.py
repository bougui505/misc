#!/usr/bin/env python3
# -*- coding: UTF8 -*-

from loader import Loader
import matplotlib.pyplot as plt
import numpy as np

name_exp = 'SAC'
save = False
show = True

model, args = Loader().load(name=name_exp)
env = args['env']

print(f"Doing prediction with a {args['model']} model")
obs = env.reset()
traj = []
total_reward = 0
while True:
    action, _states = model.predict(obs, deterministic=False)
    # action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if env.V[env.discretized_coords] == env.V.min():
        done = True
    total_reward += reward
    traj.append(env.coords.copy())
    if done:
        obs = env.reset()
        break
traj = np.asarray(traj)

if save:
    np.save('env.npy', env.V)
    np.save('traj.npy', traj)

if show:
    V = np.ma.masked_array(env.V, env.V > 200)
    plt.matshow(V, 40)
    plt.plot(traj[:, 1], traj[:, 0], color='r')
    plt.show()
