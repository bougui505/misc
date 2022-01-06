import gym
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import scipy.spatial.distance as distance


def muller_potential(x, y):
    """Muller potential
    Parameters
    ----------
    x : {float, np.ndarray}
    X coordinate. If you supply an array, x and y need to be the same shape,
    and the potential will be calculated at each (x,y pair)
    y : {float, np.ndarray}
    Y coordinate. If you supply an array, x and y need to be the same shape,
    and the potential will be calculated at each (x,y pair)
    Returns
    -------
    potential : {float, np.ndarray, or theano symbolic variable}
    Potential energy. Will be the same shape as the inputs, x and y.
    Reference
    ---------
    Code adapted from https://cims.nyu.edu/~eve2/ztsMueller.m
    """
    aa = [-1, -1, -6.5, 0.7]
    bb = [0, 0, 11, 0.6]
    cc = [-10, -10, -6.5, 0.7]
    AA = [-200, -100, -170, 15]
    XX = [1, 0, -0.5, -1]
    YY = [0, 0.5, 1.5, 1]
    # use symbolic algebra if you supply symbolic quantities
    value = 0
    for j in range(0, 4):
        value += AA[j] * np.exp(aa[j] * (x - XX[j]) ** 2 + bb[j] * (x - XX[j]) *
                                (y - YY[j]) + cc[j] * (y - YY[j]) ** 2)
    return value


def muller_mat(minx, maxx, miny, maxy, nbins, padding=None):
    grid_width = max(maxx - minx, maxy - miny) / nbins
    xx, yy = np.mgrid[minx:maxx:grid_width, miny:maxy:grid_width]
    V = muller_potential(xx, yy)
    if padding is not None:
        V = np.pad(V, pad_width=padding, constant_values=V.max())
    return V


def dijkstra(V, start, end):
    """

    :param V:
    :param start:
    :param end:
    :return:
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
        neighbors = [tuple(e) for e in np.asarray(cc) - connectivity if
                     0 <= e[0] < V.shape[0] and e[1] >= 0 and e[1] < V.shape[1]]
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


def get_potential(easy=True, padding=((0, 0), (0, 0)), use_dijkstra=False, binary=False, show=False, offset=0):
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
        start = padding[0][0], padding[1][0]
        # Use 22, the last value as a target
        x = 4
        end = x + padding[0][0], int(0.1 * x ** 2) + padding[1][0]
    else:
        minx = -1.5
        maxx = 1.2
        miny = -0.2
        maxy = 2
        V = muller_mat(minx,
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
    def __init__(self, history=2, maxiter=200, localenvshape=(36, 36), easy=False,
                 binary=True, dijkstra=False, discrete_action_space=True,
                 include_past=False, include_move=False):
        self.easy = easy
        self.binary = binary
        self.dijkstra = dijkstra
        self.history = history

        self.maxiter = maxiter
        self.history = history

        # Get the space state ready :
        self.localenvshape = localenvshape
        # This the format expected for padding : before and after, for each dimension
        self.pad = tuple((size_localenv // 2, size_localenv // 2) for size_localenv in self.localenvshape)

        # First get the raw potential as well as a local environment.
        # Pad the potential to avoid states on the border
        self.V, self.start, self.end = get_potential(show=False, easy=easy, padding=self.pad, binary=binary,
                                                     use_dijkstra=dijkstra)
        self.initial_coords = np.asarray(self.start)
        self.rewardmap_init = self.V

        # self.low high represent the i,j bounds (the inner image border)
        self.n_row, self.n_col = np.asarray(self.V.shape) - np.asarray(self.localenvshape)  # (100, 82)
        self.low = self.pad[0][0], self.pad[1][0]
        self.high = self.low + np.asarray([self.n_row, self.n_col])
        local_n_row, local_n_col = self.localenvshape
        self.history = history
        self.img_env_shape = (1, history, local_n_row, local_n_col)
        self.include_past = include_past
        self.include_move = include_move
        spacedict = {'values': gym.spaces.Box(low=self.V.min(),
                                              high=self.V.max(),
                                              shape=self.img_env_shape,
                                              dtype=np.float32)}
        if include_past:
            spacedict['past_img'] = gym.spaces.Box(low=self.V.min(),
                                                   high=self.V.max(),
                                                   shape=self.img_env_shape,
                                                   dtype=np.float32)
        self.pos_env_shape = (history, 2,)
        if include_move:
            spacedict['movement'] = gym.spaces.Box(low=-1, high=1, shape=self.pos_env_shape)
        self.observation_space = gym.spaces.Dict(spaces=spacedict)

        # Finally, set up the action space
        self.discrete_action_space = discrete_action_space
        if discrete_action_space:
            # Create a grid, then remove the middle value and shift the ones after to get all neighbors
            self.action_space = gym.spaces.Discrete(8)
            self.action_dict = {i: np.asarray((i % 3 - 1, i // 3 - 1)) for i in range(9)}
            for i in range(4, 8):
                self.action_dict[i] = self.action_dict[i + 1]
            self.action_dict.pop(8)
        else:
            self.action_space = gym.spaces.Box(low=np.asarray([-1, -1]),
                                               high=np.asarray([1, 1]),
                                               shape=(2,))

        self.reset()

    @property
    def localenv(self):
        return self.localenv_coords(self.discretized_coords)

    def localenv_coords(self, discrete_coords):
        di, dj = np.asarray(self.localenvshape, dtype=int) // 2
        i, j = discrete_coords
        return self.V[i - di:i + di, j - dj:j + dj]

    def localpast_coords(self, discrete_coords):
        # TODO : be smarter
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
        self.iter += 1
        if self.iter >= self.maxiter:
            done = True
        else:
            done = False

        # Take an action and observe the next state based on taking it.
        # Discrete
        if self.discrete_action_space:
            action = self.action_dict[action]
        # Continuous
        else:
            action /= np.linalg.norm(action)
            # # print('angle:', np.rad2deg(np.arccos(action[0])))
            # action *= np.sqrt(2)
        old_coords = self.coords
        # We clamp coords to stay in the authorized region and compute their discrete counterpart
        self.coords = self.clamp_coords(action + self.coords)
        self.discretized_coords = self.discretize_coords(self.coords)
        i1, j1 = self.discretized_coords
        self.traj.append(self.coords)
        self.state = self.state_from_traj()

        # Get the reward from the state we end up with.
        # reward = self.rewardmap[i1, j1]  # + self.milestones_reward()
        # reward = -0.1
        # reward += j1 * 0.03
        # if self.rewardmap[i1, j1] == 0:
        #     reward += 20
        reward = self.rewardmap[i1, j1]
        self.rewardmap[i1, j1] = self.rewardmap.min()

        if (i1, j1) == self.end:
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
        replay = self.traj[::-1][:self.history + 1]
        img_state = np.zeros(shape=self.img_env_shape)
        for i, step in enumerate(replay[:self.history]):
            img_state[:, -(i + 1), ...] = self.localenv_coords(step)
        spacedict = {'values': img_state}
        if self.include_past:
            past_img = np.zeros(shape=self.img_env_shape)
            for i, step in enumerate(replay[:self.history]):
                past_img[:, -(i + 1), ...] = self.localpast_coords(step)
            spacedict['past_img'] = past_img
        if self.include_move:
            pos_state = np.zeros(shape=self.pos_env_shape)
            for i, (next, prev) in enumerate(zip(replay[:self.history], replay[1:])):
                step = next - prev
                pos_state[-(i + 1), ...] = step
            spacedict['movement'] = pos_state
        return spacedict

    def reset(self, random_start=False):
        self.iter = 0

        if random_start:
            try:
                possibilities = np.asarray(np.where(self.V == self.V.max())).T
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


if __name__ == '__main__':
    pass
    from stable_baselines3.common.env_checker import check_env

    env = MullerEnv()
    check_env(env)
