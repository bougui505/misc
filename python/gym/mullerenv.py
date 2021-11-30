import gym
import numpy as np
from misc import muller_potential
from misc.box.box import Box
import scipy.spatial.distance as distance


class MullerEnv(gym.Env):
    def __init__(self, maxiter=200):
        self.traj = []
        self.maxiter = maxiter
        self.localenvshape = (36, 36)
        minx = -1.5
        maxx = 1.2
        miny = -0.2
        maxy = 2
        self.pad = np.asarray(self.localenvshape) // 2
        self.V = muller_potential.muller_mat(minx,
                                             maxx,
                                             miny,
                                             maxy,
                                             nbins=100,
                                             padding=self.pad)
        self.box = Box(self.V.shape, padding=self.pad, padded_shape=True)
        self.V[self.pad[0], :] = self.V.max()
        self.V[-(self.pad[0] + 1), :] = self.V.max()
        self.V[:, self.pad[1]] = self.V.max()
        self.V[:, -(self.pad[1] + 1)] = self.V.max()
        self.V = self.V.astype(np.float32)
        print('V.shape: ', self.V.shape)
        self.n, self.p = np.asarray(self.V.shape) - np.asarray(
            self.localenvshape)  # (100, 82)
        print('V.shape-pad: ', self.n, self.p)
        self.action_space = gym.spaces.Box(low=np.asarray([-1, -1]),
                                           high=np.asarray([1, 1]),
                                           shape=(2, ))
        low = self.pad
        high = self.pad + np.asarray([self.p, self.n])
        self.coords_space = gym.spaces.Box(low=low[::-1],
                                           high=high[::-1],
                                           shape=(2, ))
        n, p = self.localenvshape
        self.observation_space = gym.spaces.Box(low=self.V.min(),
                                                high=self.V.max(),
                                                shape=(1, n, p),
                                                dtype=np.float32)
        # self.j_stop, self.i_stop = np.unravel_index(self.V.argmin(),
        #                                             self.V.shape)

    @property
    def localenv(self):
        # localenv = self.V.copy()
        # if len(self.traj) > 0:
        #     inds = np.int_(np.round(np.asarray(self.traj)))
        #     inds = (inds[:, 0], inds[:, 1])
        #     localenv[inds] = 2
        di, dj = np.asarray(self.localenvshape, dtype=int) // 2
        i, j = self.discretized_coords
        return self.V[i - di:i + di, j - dj:j + dj]

    @property
    def discretized_coords(self):
        i, j = np.int_(np.round(self.coords))
        i, j = self.box.bounding_coords((i, j), padded=True)
        return (i, j)

    @property
    def excluded_volume(self):
        sigma = 1.
        traj = np.asarray(self.traj)
        cdist = distance.cdist(XA=traj[-1][None, ...], XB=traj[:-1])
        out = np.exp(-cdist**2 / (2 * sigma**2))
        return out.sum()

    def step(self, action):
        done = False
        win = False
        # action = 2 * action / np.linalg.norm(action)
        self.iter += 1
        # if self.iter >= self.maxiter:
        #     done = True
        # else:
        #     done = False
        ind_prev = np.copy(self.discretized_coords)
        self.coords += action
        # if not self.coords_space.contains(self.coords):
        #     self.coords = coords_prev
        #     loose = True
        #     done = True
        i0, j0 = ind_prev
        i1, j1 = self.discretized_coords
        if self.box.bounded:
            done = True
        self.traj.append(self.coords)
        if self.V[i1, j1] <= -130.:
            # if self.V[i1, j1] == self.V.min():
            win = True
            done = True
        # reward = -np.exp(0.01 * (self.V[i1, j1] - self.V[i0, j0]))
        # reward = np.exp(-0.01 * self.V[i1, j1])
        reward = -np.exp(-4. * (1. - self.V[i1, j1] / self.V.max()))
        if win:
            reward = 100.
        self.state = self.localenv[None, ...]
        i, j = self.discretized_coords
        # print(self.iter, i, j, self.i_stop, self.j_stop)
        # if done is None:
        #     if (i, j) == (self.i_stop, self.j_stop):
        #         done = True
        #         # reward = 100000.
        #     else:
        #         done = False
        info = {}
        if done:
            print('iter:', self.iter)
            print('pos:', i, j)
            # if win:
            #     print('win')
            # if loose:
            #     print('loose')
            # print('stop', self.i_stop, self.j_stop)
        # print(self.localenv.shape)
        # reward -= self.excluded_volume
        return self.state, float(reward), done, info

    def reset(self):
        # self.coords = self.coords_space.sample()
        self.coords = np.asarray([98., 27.])
        self.state = self.localenv[None, ...]
        self.iter = 0
        self.traj = []
        return self.state
