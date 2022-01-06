#!/usr/bin/env python3
# -*- coding: UTF8 -*-

from stable_baselines3 import A2C, SAC
from mullerenv import MullerEnv
from feature_extractor import CustomCNN
from loader import Loader

dump_name = 'SAC'
history = 2
maxiter = 200
localenvshape = (36, 36)
easy = True
binary = False
dijkstra = False
discrete_action_space = False
include_traj = True
include_move = True
model_type = 'SAC'
total_timesteps = 500
log_interval = 20

env = MullerEnv(history=history, maxiter=maxiter, localenvshape=localenvshape, easy=easy,
                binary=binary, dijkstra=dijkstra, discrete_action_space=discrete_action_space,
                include_past=include_traj, include_move=include_move)
features_extractor_kwargs = dict(history=history, discrete_action_space=discrete_action_space,
                                 include_traj=include_traj, include_move=include_move)

# We need to distinguish based on the network names in sb3
model_type = model_type.lower()
if model_type == 'a2c':
    policy_kwargs = dict(net_arch=[dict(vf=[24], pi=[24])],
                         features_extractor_class=CustomCNN,
                         features_extractor_kwargs=features_extractor_kwargs)
    model = A2C("MultiInputPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
elif model_type == 'sac':
    assert not discrete_action_space
    policy_kwargs = dict(features_extractor_class=CustomCNN,
                         net_arch=dict(qf=[24], pi=[24]),
                         features_extractor_kwargs=features_extractor_kwargs, )
    model = SAC("MultiInputPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
else:
    raise NotImplementedError

model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
loader = Loader(model=model, args={'model': model_type, 'policy_kwargs': policy_kwargs, 'env': env})
loader.save(dump_name)
