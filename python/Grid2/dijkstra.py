#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2021 Institut Pasteur                                       #
#                 				                            #
#                                                                           #
#  Redistribution and use in source and binary forms, with or without       #
#  modification, are permitted provided that the following conditions       #
#  are met:                                                                 #
#                                                                           #
#  1. Redistributions of source code must retain the above copyright        #
#  notice, this list of conditions and the following disclaimer.            #
#  2. Redistributions in binary form must reproduce the above copyright     #
#  notice, this list of conditions and the following disclaimer in the      #
#  documentation and/or other materials provided with the distribution.     #
#  3. Neither the name of the copyright holder nor the names of its         #
#  contributors may be used to endorse or promote products derived from     #
#  this software without specific prior written permission.                 #
#                                                                           #
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS      #
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT        #
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR    #
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT     #
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,   #
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT         #
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    #
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY    #
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT      #
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE    #
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.     #
#                                                                           #
#  This program is free software: you can redistribute it and/or modify     #
#                                                                           #
#############################################################################

import numpy as np


def dijkstra(V):
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
    cc = np.unravel_index(V.argmin(), m.shape)  # current_cell
    m[cc] = 0
    P = {}  # dictionary of predecessors
    # while (~visit_mask).sum() > 0:
    for _ in range(V.size):
        # print cc
        neighbors = [
            tuple(e) for e in np.asarray(cc) - connectivity if e[0] > 0
            and e[1] > 0 and e[0] < V.shape[0] and e[1] < V.shape[1]
        ]
        neighbors = [e for e in neighbors if not visit_mask[e]]
        tentative_distance = np.asarray([V[e] - V[cc] for e in neighbors])
        for i, e in enumerate(neighbors):
            d = tentative_distance[i] + m[cc]
            if d < m[e]:
                m[e] = d
                P[e] = cc
        visit_mask[cc] = True
        m_mask = np.ma.masked_array(m, visit_mask)
        cc = np.unravel_index(m_mask.argmin(), m.shape)
    return m, P


def shortestPath(start, end, P):
    """

    Args:
        start: starting cell
        end: ending cell
        P: dictionary of predecessors

    Returns:
        Path

    """
    Path = []
    step = end
    while 1:
        Path.append(step)
        if step == start:
            break
        step = P[step]
    Path.reverse()
    return np.asarray(Path)


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-a', '--arg1')
    args = parser.parse_args()

    import misc.muller_potential
    import matplotlib.pyplot as plt

    V = misc.muller_potential.muller_mat(minx=-1.5,
                                         maxx=1.2,
                                         miny=-0.2,
                                         maxy=2,
                                         nbins=100,
                                         padding=18)
    D, P = dijkstra(V)
    V = np.ma.masked_array(V, V > 200)
    path = shortestPath(np.unravel_index(V.argmin(), V.shape), (98, 27), P)
    plt.contourf(V, 40)
    plt.plot(path[:, 1], path[:, 0], 'r.-')
    plt.show()
