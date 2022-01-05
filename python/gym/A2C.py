#!/usr/bin/env python3
# -*- coding: UTF8 -*-

from stable_baselines3 import A2C
from mullerenv import MullerEnv
from feature_extractor import CustomCNN

history = 1

env = MullerEnv(history=history)
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    net_arch=[dict(vf=[24], pi=[24])],
    # features_extractor_kwargs=dict(features_dim=8),
)

model = A2C(
    "MultiInputPolicy",
    env,
    verbose=1,
    policy_kwargs=policy_kwargs,
)
model.learn(total_timesteps=50000, log_interval=100)
model.save("A2C_muller")
