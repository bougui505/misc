import sys

import gym
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import scipy.spatial.distance as distance
import torch

from misc import muller_potential
from misc.box.box import Box


def dijkstra(V, start, end):
    """

    Args:
        V:

    Returns:
        m: mask
        P: dictionary of predecessors

    """
    V = np.ma.masked_array(V, np.zeros(V.shape, dtype=bool))
    mask = V.mask
    visit_mask = mask.copy()  # mask visited cells
    m = np.ones_like(V) * np.inf
    connectivity = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1]
                    if (not (i == j == 0))]
    cc = end  # current_cell
    m[cc] = 0
    pred = {}  # dictionary of predecessors

    break_next = False
    for _ in range(V.size):
        # Get neighbors list
        neighbors = [tuple(e) for e in np.asarray(cc) - connectivity if e[0] >= 0
                     and e[1] >= 0 and e[0] < V.shape[0] and e[1] < V.shape[1]]
        neighbors = [e for e in neighbors if not visit_mask[e]]
        tentative_distance = np.asarray([V[e] for e in neighbors])
        # tentative_distance = np.asarray([V[e] - V[cc] for e in neighbors])
        for i, e in enumerate(neighbors):
            d = tentative_distance[i] + m[cc]
            if d < m[e]:
                m[e] = d
                pred[e] = cc
        visit_mask[cc] = True
        m_mask = np.ma.masked_array(m, visit_mask)
        cc = np.unravel_index(m_mask.argmin(), m.shape)
        if tuple(cc) == tuple(start):
            break_next = True
        if break_next:
            break

    # Then unroll this path
    path = []
    step = start
    while 1:
        path.append(step)
        if step == end:
            break
        step = pred[step]
    return np.asarray(path)


def get_potential(easy=True, padding=(0, 0), use_dijkstra=False, binary=False, show=False, offset=0):
    """
    We always return a reward-like matrix, suitably padded.
    The maximum values are around 1 and the negative ones are close to zero.

    We offer two possible potentials, easy (follow a quadratic function) and hard (muller).
    We also offer the possibility to use simply the beginning and end of the potentials to just play a maze game.

    For all these possibilities, we offer an optimal solution obtained using Djikstra, and then we would return an
    edt from this optimal path.
    :param easy:
    :param padding:
    :param binary:
    :param use_dijkstra:
    :param show:
    :return:
    """
    if easy:
        V = np.ones((50, 50))
        for x in range(len(V)):
            y = int(0.1 * x ** 2)
            if y < len(V):
                V[x, y] = 0
        V = ndimage.distance_transform_edt(V)
        V = np.exp(-V / 10)
        V = np.pad(V, pad_width=padding, constant_values=V.min())
        start = tuple(padding)
        # Use 22, the last value as a target
        x = 22
        end = x + padding[0], int(0.1 * x ** 2) + padding[1]
    else:
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
        V = np.exp(-(V - V.min()) / 100)
        # Define a random starting point and find the argmin (53, 79)
        start, end = (98, 27), np.unravel_index(V.argmax(), V.shape)

    # This could be more efficient, but we did it afterwards and it's nothing
    if binary:
        V = 0 * np.ones_like(V)
        V[start] = 1
        V[end] = 1

    # Then Use Djikstra to get an idea of the optimum cost minimizing this route
    if use_dijkstra:
        # For djikstra we need the reverse : the distance to the minimum (and we add epsilon)
        # In the easy setting, we get diagonals so conter-intuitive results
        V = 0.01 + V.max() - V
        path = dijkstra(V, start, end)
        rewardmap = np.ones_like(V)
        rewardmap[tuple(path.T)] = 0
        V = ndimage.distance_transform_edt(rewardmap)
        V = np.exp(-V / 10)

    V += offset

    if show:
        # We need to reverse the values to plot y as a function of x (and use scatter...)
        plt.matshow(V.T, origin='lower')
        plt.scatter(*start)
        plt.scatter(*end)
        if use_dijkstra:
            plt.plot(*tuple(path.T), c='r')
        plt.colorbar()
        plt.show()
    return V, start, end


class MullerEnv(gym.Env):
    def __init__(self, history=2, maxiter=200, localenvshape=(36, 36), easy=True,
                 binary=True, dijkstra=True, discrete_action_space=True):
        self.easy = easy
        self.binary = binary
        self.dijkstra = dijkstra
        self.history = history

        self.maxiter = maxiter
        self.history = history

        # Get the space state ready :
        # First get the raw potential as well as a local environment.
        # Pad the potential to avoid states on the border
        self.localenvshape = localenvshape
        self.pad = np.asarray(self.localenvshape) // 2
        self.V, self.start, self.end = get_potential(show=True, easy=easy, padding=self.pad, binary=binary,
                                                     use_dijkstra=dijkstra)

        self.n, self.p = np.asarray(self.V.shape) - np.asarray(self.localenvshape)  # (100, 82)
        self.initial_coords = np.asarray(self.start)
        self.rewardmap_init = self.V

        print(self.V.max())
        print(self.V.min())
        sys.exit()

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
            'traj_img': gym.spaces.Box(low=self.V.min(),
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

    def localpast_coords(self, discrete_coords):
        di, dj = np.asarray(self.localenvshape, dtype=int) // 2
        i, j = discrete_coords
        # localpast_img = np.zeros(self.localenvshape)

        globalpast_img = np.zeros_like(self.V)
        globalpast_img[tuple(np.asarray(self.traj).T)] = 1

        return globalpast_img[i - di:i + di, j - dj:j + dj]

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

        # reward = self.rewardmap[i1, j1]  # + self.milestones_reward()
        self.rewardmap[i1, j1] = self.rewardmap.min()
        reward = -0.1

        # reward += j1 * 0.03
        if self.rewardmap[i1, j1] == 0:
            reward += 20
            print('toto')
            # print(self.rewardmap.mean())
            # print((self.rewardmap == 0.).sum())
            # print(self.rewardmap.min())
            # print(reward)

        if (i1, j1) == self.end:
            win = True
            done = True
            reward += 20

        info = {}
        # if done:
        #     print('iter:', self.iter)
        #     print('pos:', i1, j1)
        return self.state, float(reward), done, info

    def state_from_traj(self):
        """
        Generate a state observation from the trajectory.
        """
        img_state = np.zeros(shape=self.img_env_shape)
        traj_img = np.zeros(shape=self.img_env_shape)
        pos_state = np.zeros(shape=self.pos_env_shape)
        replay = self.traj[::-1][:self.history]
        for i, step in enumerate(replay):
            img_state[:, -(i + 1), ...] = self.localenv_coords(step)
            traj_img[:, -(i + 1), ...] = self.localpast_coords(step)
            pos_state[-(i + 1), ...] = step
        return {'values': img_state, 'movement': pos_state, 'traj_img': traj_img}

    def reset(self, random_start=False):
        self.iter = 0

        if random_start:
            try:
                possibilities = np.asarray(np.where(self.V == self.V.max())).T
                print(possibilities)
                chosen_idx = np.random.choice(len(possibilities))
                self.coords = possibilities[chosen_idx]
            except ValueError:
                self.coords = self.initial_coords
        else:
            self.coords = self.initial_coords
        self.discretized_coords = self.discretize_coords(self.coords)
        self.traj = [self.discretized_coords]

        self.rewardmap = self.rewardmap_init.copy()
        # self.state = self.localenv[None, ...]
        self.state = self.state_from_traj()

        return self.state
