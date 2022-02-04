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
from sklearn.cluster import AgglomerativeClustering


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


def get_distances(timeseries):
    npts = len(timeseries)
    distances = (timeseries[1:] - timeseries[:-1])**2
    pmat = np.ones((npts, npts)) * 9999.
    np.fill_diagonal(pmat, 0.)
    pmat[np.arange(npts - 1), np.arange(1, npts)] = distances
    pmat[np.arange(1, npts), np.arange(npts - 1)] = distances
    return pmat


def get_clusters(timeseries, ninter):
    pmat = get_distances(timeseries)
    agg = AgglomerativeClustering(n_clusters=ninter,
                                  affinity='precomputed',
                                  linkage='average')
    labels = agg.fit_predict(pmat)
    return labels


if __name__ == '__main__':
    import argparse
    from misc.interpolation import read_stdin, parse_fields, format_output
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-d', '--delimiter', help='Delimiter to use', type=str)
    parser.add_argument(
        '-f',
        '--fields',
        help='Fields (default: xy). Give the column field names',
        default='xy')
    parser.add_argument('--ninter',
                        help='Number of intervals',
                        type=int,
                        required=True)
    args = parser.parse_args()

    A = read_stdin(delimiter=args.delimiter)
    fields = parse_fields(args.fields)
    x, y = A[:, fields == 'x'], A[:, fields == 'y']
    y = np.squeeze(y)
    if x.shape[1] == 1:
        x = np.squeeze(x)
    else:
        print('Multiple x value given. Not yet implemented')
        sys.exit()
    labels = get_clusters(y, args.ninter)
    format_output(x, y, delimiter=args.delimiter, label=labels)
