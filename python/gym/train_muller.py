#!/usr/bin/env python3
# -*- coding: UTF8 -*-

from stable_baselines3 import A2C, SAC
from mullerenv import MullerEnv
from feature_extractor import CustomCNN
from loader import Loader
import argparse

localenvshape = (36, 36)
log_interval = 100

parser = argparse.ArgumentParser(description='')
parser.add_argument('-n', '--dump_name', help='Name used for saving', type=str, default=None)
parser.add_argument('--history', help='Number of steps kept in memory', type=int, default=2)
parser.add_argument('-mi', '--max_iter', help='Maximal number of steps before stopping the episode', type=int,
                    default=200)
parser.add_argument('--discrete', help='Use discrete space action', action='store_true', default=False)
parser.add_argument('--easy', help='Easy setting of quadratic instead of muller', action='store_true', default=False)
parser.add_argument('--binary', help='No continuous potential, just binary', action='store_true', default=False)
parser.add_argument('--djikstra', help='Use the optimal path instead of the raw field', action='store_true',
                    default=False)
parser.add_argument('--include_traj', help='Add the previous states in the state', action='store_true', default=False)
parser.add_argument('--include_move', help='Add the previous move in the state', action='store_true', default=False)
parser.add_argument('--model_type', help='Kind of model in use, SAC or A2C', type=str, default=None)
parser.add_argument('--total_timesteps', help='Maximal number of steps before stopping the episode', type=int,
                    default=5000)
# parser.add_argument('--anchor', nargs=3, metavar=('x', 'y', 'z'), type=float,
#                     help='xyz coordinates of the center of the blob to monitor. Can be pasted from the pymol plugin.')
args = parser.parse_args()

print(args)

env = MullerEnv(history=args.history, maxiter=args.max_iter, localenvshape=localenvshape, easy=args.easy,
                binary=args.binary, dijkstra=args.djikstra, discrete_action_space=args.discrete,
                include_past=args.include_traj, include_move=args.include_move)
features_extractor_kwargs = dict(history=args.history, discrete_action_space=args.discrete,
                                 include_traj=args.include_traj, include_move=args.include_move)

# We need to distinguish based on the network names in sb3
model_type = args.model_type.lower()
if model_type == 'a2c':
    policy_kwargs = dict(net_arch=[dict(vf=[24], pi=[24])],
                         features_extractor_class=CustomCNN,
                         features_extractor_kwargs=features_extractor_kwargs)
    model = A2C("MultiInputPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
elif model_type == 'sac':
    assert not args.discrete_action_space
    policy_kwargs = dict(features_extractor_class=CustomCNN,
                         net_arch=dict(qf=[24], pi=[24]),
                         features_extractor_kwargs=features_extractor_kwargs, )
    model = SAC("MultiInputPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
else:
    raise NotImplementedError

model.learn(total_timesteps=args.total_timesteps, log_interval=log_interval)
loader = Loader(model=model, args={'model': model_type, 'policy_kwargs': policy_kwargs, 'env': env})
loader.save(args.dump_name)
