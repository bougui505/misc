import sys

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from misc import muller_potential
from misc.box.box import Box
import scipy.spatial.distance as distance
from misc.gym import dijkstra


def get_potential(padding):
    minx = -1.5
    maxx = 1.2
    miny = -0.2
    maxy = 2
    V = muller_potential.muller_mat(minx,
                                    maxx,
                                    miny,
                                    maxy,
                                    nbins=100,
                                    padding=padding)
    return V


def get_potential_easy():
    V = np.ones((50, 50))
    for x in range(len(V)):
        y = int(0.1 * x ** 2)
        if y < len(V):
            V[x, y] = 0

    from scipy import ndimage
    V = ndimage.distance_transform_edt(V)
    V = np.exp(-V / 10)
    return V


class MullerEnv(gym.Env):
    def __init__(self, history=2):
        # Get the space state ready :
        # First get the raw potential as well as a local environment.
        # Pad the potential to avoid states on the border
        self.localenvshape = (36, 36)
        self.pad = np.asarray(self.localenvshape) // 2
        V = get_potential_easy()
        self.V = np.pad(V, self.pad, constant_values=V.min()) - 1
        self.start = (18, 18)
        x = 10
        self.end = x + 18, int(0.1 * x ** 2) + 18

        self.rewardmap_init = self.V

        # plt.matshow(self.V)
        # plt.colorbar()
        # plt.show()
        # sys.exit()

        # V = get_potential(padding=self.pad)

        # Define a random starting point and find the argmin (53, 79)
        # Then Use Djikstra to get an idea of the optimum cost minimizing this route
        # start, end = np.unravel_index(V.argmin(), V.shape), (98, 27)
        # self.coords = np.asarray([98., 27.])
        # print(start)
        # print(end)
        self.initial_coords = np.asarray([18., 18.])
        self.maxiter = 200

        # rewardmap_init, self.path = dijkstra.discriminator(V=V,
        #                                                    start=start,
        #                                                    end=end)
        # rewardmap_init = np.clip(rewardmap_init, a_min=-1, a_max=0)
        # self.rewardmap_init = -np.sqrt(-rewardmap_init)
        # self.path = [tuple(e) for e in self.path]
        # self.pathlen = len(self.path)
        # self.maxiter = self.pathlen

        # Use the reward map from djikstra to converge to this minimum.
        # self.rewardmap_init -= self.rewardmap_init.min()
        # self.rewardmap = self.rewardmap_init.copy()

        # Finally,
        # absmax = max(V.min(), V.max(), key=abs)
        # self.V = 1 - V.astype(np.float32) / (absmax)
        self.n, self.p = np.asarray(self.V.shape) - np.asarray(self.localenvshape)  # (100, 82)
        # print('V.shape-pad: ', self.n, self.p)

        # self.action_space = gym.spaces.Box(low=np.asarray([-1, -1]),
        #                                    high=np.asarray([1, 1]),
        #                                    shape=(2,))

        self.action_space = gym.spaces.Discrete(8)

        # Create a grid, then remove the middle value and shift the ones after to get all neighbors
        self.action_dict = {i: np.asarray((i % 3 - 1, i // 3 - 1)) for i in range(9)}
        for i in range(4, 8):
            self.action_dict[i] = self.action_dict[i + 1]
        self.action_dict.pop(8)

        low = self.pad
        high = self.pad + np.asarray([self.p, self.n])
        # self.low high represent the i,j bounds
        self.low, self.high = low[::-1], high[::-1]
        self.coords_space = gym.spaces.Box(low=low[::-1],
                                           high=high[::-1],
                                           shape=(2,))
        local_n, local_p = self.localenvshape
        # self.observation_space = gym.spaces.Box(low=self.V.min(),
        #                                         high=self.V.max(),
        #                                         shape=(1, local_n, local_p),
        #                                         dtype=np.float32)
        self.history = history
        self.img_env_shape = (1, history, local_n, local_p)
        self.pos_env_shape = (history, 2,)
        self.observation_space = gym.spaces.Dict({
            'values': gym.spaces.Box(low=self.V.min(),
                                     high=self.V.max(),
                                     shape=self.img_env_shape,
                                     dtype=np.float32),
            'movement': gym.spaces.Box(low=-1, high=1, shape=self.pos_env_shape), })

        self.reset()

    @property
    def localenv(self):
        return self.localenv_coords(self.discretized_coords)

    def localenv_coords(self, discrete_coords):
        di, dj = np.asarray(self.localenvshape, dtype=int) // 2
        i, j = discrete_coords
        return self.V[i - di:i + di, j - dj:j + dj]

    def discretize_coords(self, coords):
        i, j = np.int_(np.round(coords))
        return (i, j)

    def clamp_coords(self, coords):
        i, j = np.int_(np.round(coords))
        thresh_i = min(max(self.low[0], i), self.high[0])
        thresh_j = min(max(self.low[1], j), self.high[1])
        return np.asarray((thresh_i, thresh_j))

    @property
    def excluded_volume(self):
        sigma = 1.
        traj = np.asarray(self.traj)
        cdist = distance.cdist(XA=traj[-1][None, ...], XB=traj[:-1])
        out = np.exp(-cdist ** 2 / (2 * sigma ** 2))
        return out.sum()

    def milestones_reward(self):
        if tuple(self.discretized_coords) in self.path[::10]:
            return 10.
        else:
            return 0.

    def step(self, action):
        # We always do this computation
        done = False
        win = False
        self.iter += 1
        if self.iter >= self.maxiter:
            done = True
        else:
            done = False

        # What does the action do ?
        # # Continuous !
        # action /= np.linalg.norm(action)
        # # print('angle:', np.rad2deg(np.arccos(action[0])))
        # action *= np.sqrt(2)
        # # action = 2 * action / np.linalg.norm(action)

        # Discrete :
        action = self.action_dict[action]

        # We clamp coords to stay in the authorized region and compute their discrete counterpart
        old_coords = self.coords
        self.coords = self.clamp_coords(action + self.coords)
        self.discretized_coords = self.discretize_coords(self.coords)
        i1, j1 = self.discretized_coords
        self.traj.append(self.coords)
        movement = self.coords - old_coords
        # self.state = {'values': self.localenv[None, ...], 'movement': movement}
        # self.state = self.localenv[None, ...]
        self.state = self.state_from_traj()

        reward = self.rewardmap[i1, j1]  # + self.milestones_reward()
        self.rewardmap[i1, j1] = self.rewardmap.min()

        # reward += j1 * 0.03

        if (i1, j1) == self.end:
            win = True
            done = True
            reward += 20

        info = {}
        if done:
            print('iter:', self.iter)
            print('pos:', i1, j1)
        return self.state, float(reward), done, info

    def state_from_traj(self):
        img_state = np.zeros(shape=self.img_env_shape)
        pos_state = np.zeros(shape=self.pos_env_shape)
        replay = self.traj[::-1][:self.history]
        for i, step in enumerate(replay):
            img_state[:, -(i + 1), ...] = self.localenv_coords(step)
            pos_state[-(i + 1), ...] = step
        return {'values': img_state, 'movement': pos_state}

    def reset(self):
        self.iter = 0
        self.coords = self.initial_coords
        self.discretized_coords = self.discretize_coords(self.coords)
        self.traj = [self.discretized_coords]

        self.rewardmap = self.rewardmap_init.copy()
        # self.state = self.localenv[None, ...]
        self.state = self.state_from_traj()

        return self.state
