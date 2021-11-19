#!/usr/bin/env python3
# -*- coding: UTF8 -*-

import numpy as np
import matplotlib.pyplot as plt

V = np.load('env.npy')
traj = np.load('traj.npy')

V = np.ma.masked_array(V, V > 200)
plt.contourf(V, 40)

plt.plot(traj[:, 0], traj[:, 1], color='r')

plt.show()
