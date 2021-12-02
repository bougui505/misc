#!/usr/bin/env python3
# -*- coding: UTF8 -*-

import gym
from stable_baselines3 import SAC
from mullerenv import MullerEnv
from feature_extractor import CustomCNN

env = MullerEnv()

policy_kwargs = dict(features_extractor_class=CustomCNN,
                     # features_extractor_kwargs=dict(features_dim=8),
                     )

model = SAC(
    "CnnPolicy",
    env,
    verbose=1,
    train_freq=(1, "episode"),
    # ent_coef=0.1,
    policy_kwargs=policy_kwargs,
    use_sde=False,
    sde_sample_freq=100)
model.learn(total_timesteps=250000, log_interval=1)
model.save("sac_muller")
