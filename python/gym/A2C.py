#!/usr/bin/env python3
# -*- coding: UTF8 -*-

from stable_baselines3 import A2C
from mullerenv import MullerEnv
from feature_extractor import CustomCNN

history = 3

env = MullerEnv(history=history)
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    net_arch=[dict(vf=[24], pi=[24])],
    # features_extractor_kwargs=dict(features_dim=8),
)


def lr_sched(x, y0=1e-3, yf=1e-6):
    """
    x: current progress remaining (from 1 to 0)
    """
    y = (y0 - yf) * x + yf
    return y


model = A2C(
    "MultiInputPolicy",
    # "CnnPolicy",
    env,
    verbose=1,
    # ent_coef=0.1,
    policy_kwargs=policy_kwargs,
    use_sde=False,
    sde_sample_freq=-1,
    learning_rate=lr_sched,
    # buffer_size=100,
    # batch_size=8
)
model.learn(total_timesteps=20000, log_interval=100)
model.save("sac_muller")
