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

import sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize, NonlinearConstraint


def random_timeseries(npts):
    """
    Generate a random timeseries with npts points
    """
    timeseries = np.random.uniform(size=npts)
    timeseries[:100] += 50.
    timeseries[500:] -= 10.
    return timeseries


def interpolate(timeseries, kind='linear'):
    npts = len(timeseries)
    if timeseries.ndim == 1:
        x = np.arange(npts)
        y = timeseries
    else:
        x, y = timeseries.T
    f = interp1d(x, y, kind=kind)
    return f


def split(timeseries, intervals, delta_t):
    """
    Split the timeseries in the given intervals [(t0, t1), (t2, t3), ...]
    """
    f = interpolate(timeseries)
    npts = len(timeseries)
    subtimeseries = []
    for interval in intervals:
        t0, t1 = interval
        trange = np.arange(t0, t1, step=delta_t)
        trange = trange[trange < npts - 1]
        ts = f(trange)
        subtimeseries.append(ts)
    return subtimeseries


def std(ts, mean=False):
    """
    Return the standard deviation of a timeseries or a list of timeseries
    """
    if not hasattr(ts[0], '__iter__'):
        ts = [
            ts,
        ]
    stds = []
    for _ts_ in ts:
        stds.append(np.std(_ts_))
    stds = np.asarray(stds)
    if not mean:
        return stds
    else:
        return stds.mean()


def get_intervals(tlist, npts):
    tlist = list(tlist)
    tlist.sort()
    left = [
        0,
    ] + tlist
    right = tlist + [
        npts,
    ]
    intervals = list(zip(left, right))
    return intervals


class Objective(object):
    def __init__(self, timeseries, nsplit, delta_t):
        self.timeseries = timeseries
        self.nsplit = nsplit
        self.delta_t = delta_t
        self.npts = len(timeseries)

    def func(self, tlist):
        intervals = get_intervals(tlist, self.npts)
        splitted = split(self.timeseries,
                         intervals=intervals,
                         delta_t=self.delta_t)
        return std(splitted, mean=True)


def optimal_split(timeseries, nsplit, delta_t=.1):
    objective = Objective(timeseries=timeseries,
                          nsplit=nsplit,
                          delta_t=delta_t)
    npts = len(timeseries)
    interval_len = npts // nsplit
    t0 = [interval_len + i * interval_len for i in range(nsplit - 1)]
    # bounds = [
    #     (0, npts - 1),
    # ] * (nsplit - 1)
    res = minimize(objective.func,
                   t0,
                   method='Nelder-Mead',
                   options={'disp': True})
    tlist = res.x
    print(tlist)
    intervals = get_intervals(tlist, npts)
    splitted = split(timeseries, intervals, delta_t)
    print(std(splitted))


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--test', help='Test the module', action='store_true')
    parser.add_argument('--npts',
                        help='Test the module. Number of points for the test',
                        type=int,
                        default=1000)
    args = parser.parse_args()

    if args.test:
        timeseries = random_timeseries(args.npts)
        optimal_split(timeseries, nsplit=10)
