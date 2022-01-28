#!/usr/bin/env python3
# -*- coding: UTF8 -*-

from loader import Loader
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('-n', '--name_exp', help='Name used for saving', type=str, default=None)
parser.add_argument('-d', '--deterministic', help='No continuous potential, just binary', action='store_true',
                    default=False)
# parser.add_argument('--anchor', nargs=3, metavar=('x', 'y', 'z'), type=float,
#                     help='xyz coordinates of the center of the blob to monitor. Can be pasted from the pymol plugin.')
args = parser.parse_args()

save = False
show = True

model, exp_args = Loader().load(name=args.name_exp)
env = exp_args['env']

print(f"Doing prediction with a {exp_args['model']} model")
obs = env.reset()
traj = []
total_reward = 0
while True:
    action, _states = model.predict(obs, deterministic=args.deterministic)
    # action = int(action)
    # action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    # print(obs['values'].sum())
    # print(env.traj[-5:])
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
