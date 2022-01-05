#!/usr/bin/env python3
# -*- coding: UTF8 -*-

from stable_baselines3 import A2C
from mullerenv import MullerEnv
from feature_extractor import CustomCNN

history = 2
maxiter = 200
localenvshape = (36, 36)
easy = False
binary = True
dijkstra = False
discrete_action_space = True,
include_traj = True
include_move = False

env = MullerEnv(history=history, maxiter=maxiter, localenvshape=localenvshape, easy=easy,
                binary=binary, dijkstra=dijkstra, discrete_action_space=discrete_action_space,
                include_past=include_traj, include_move=include_move)
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    net_arch=[dict(vf=[24], pi=[24])],
    features_extractor_kwargs=dict(history=history, discrete_action_space=discrete_action_space,
                                   include_traj=include_traj, include_move=include_move),
)

model = A2C(
    "MultiInputPolicy",
    env,
    verbose=1,
    policy_kwargs=policy_kwargs,
)
model.learn(total_timesteps=50000, log_interval=100)
model.save("A2C_muller")
