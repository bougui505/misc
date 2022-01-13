#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2022 Institut Pasteur                                       #
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

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def potential(x, xdeep=-3, xshal=3, wellslope=2):
    y = ((x - xdeep) * (x - xshal))**2 + wellslope * x
    return y


def plot_potential(pos=-3):
    x = np.linspace(-5, 5, 100)
    y = potential(x)
    plt.plot(x, y)
    if pos is not None:
        plt.scatter(pos, potential(pos), marker='o', color='r')


def move(x, beta=0.1):
    dx = np.random.uniform(low=-.5, high=.5)
    x_new = x + dx
    V = potential(x)
    V_new = potential(x_new)
    delta = V_new - V
    p = np.exp(-beta * delta)
    if p > np.random.uniform():
        return x_new, V_new
    else:
        return x, V


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-a', '--arg1')
    args = parser.parse_args()

    fig, ax = plt.subplots()
    xs = np.linspace(-5, 5, 100)
    ys = potential(xs)
    ax.plot(xs, ys)
    x = -3
    dot, = ax.plot(x, potential(x), marker='o', color='red')
    if not os.path.exists('MCmovie'):
        os.mkdir('MCmovie')
        plt.savefig(f'MCmovie/{0:04d}.png')
    for i in range(100):
        x, V = move(x)
        dot.remove()
        dot, = ax.plot(x, V, marker='o', color='red')
        plt.savefig(f'MCmovie/{i+1:04d}.png')
        print(x, V)
