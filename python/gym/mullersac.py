#!/usr/bin/env python3
# -*- coding: UTF8 -*-

import gym
from stable_baselines3 import SAC
from mullerenv import MullerEnv

env = MullerEnv()

model = SAC("CnnPolicy", env, verbose=1, train_freq=(1, "episode"), ent_coef=0.1)
model.learn(total_timesteps=400000, log_interval=1)
model.save("sac_muller")
