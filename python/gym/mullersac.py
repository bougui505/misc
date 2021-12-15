#!/usr/bin/env python3
# -*- coding: UTF8 -*-

import gym
from stable_baselines3 import SAC
from mullerenv import MullerEnv
from feature_extractor import CustomCNN

env = MullerEnv()

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    net_arch=dict(qf=[512, 512], pi=[512, 512]),
    # features_extractor_kwargs=dict(features_dim=8),
)


def lr_sched(x, y0=1e-3, yf=1e-6):
    """
    x: current progress remaining (from 1 to 0)
    """
    y = (y0 - yf) * x + yf
    return y


model = SAC(
    "CnnPolicy",
    env,
    verbose=1,
    train_freq=(1, "episode"),
    # ent_coef=0.1,
    policy_kwargs=policy_kwargs,
    use_sde=False,
    sde_sample_freq=-1,
    learning_rate=lr_sched,
    # buffer_size=100,
    # batch_size=8
)
model.learn(total_timesteps=250000, log_interval=1)
model.save("sac_muller")
