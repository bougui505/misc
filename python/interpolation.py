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


def format_line(line, delimiter):
    line = line.split(delimiter)
    outline = []
    for e in line:
        e = e.strip()
        try:
            e = int(float(e)) if int(float(e)) == float(e) else float(e)
        except ValueError:
            pass
        outline.append(e)
    return outline


def read_stdin(delimiter=None):
    A = []
    with sys.stdin as inpipe:
        for line in inpipe:
            line = format_line(line, delimiter)
            A.append(line)
    A = np.asarray(A, dtype=float)
    return A


def interpolate(x, y, step, xmin=None, xmax=None, kind='linear'):
    f = interp1d(x, y, kind=kind)
    if xmin is None:
        xmin = x.min()
    if xmax is None:
        xmax = x.max()
    x = np.arange(xmin, xmax, step)
    y = f(x)
    return x, y


def format_output(x, y, delimiter):
    if delimiter is None:
        delimiter = ' '
    np.savetxt(sys.stdout, np.c_[x, y], delimiter=delimiter, fmt='%.4g')


def parse_fields(fields):
    fields = np.asarray([fields[i] for i in range(len(fields))], dtype=str)
    return fields


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-s',
                        '--step',
                        help='Spacing between values',
                        type=float,
                        required=True)
    parser.add_argument(
        '-m',
        '--min',
        help=
        'x-min value to start the interpolation with. By default, the min x-values is used',
        type=float,
        required=False)
    parser.add_argument(
        '-M',
        '--max',
        help=
        'x-max value to start the interpolation with. By default, the max x-values is used',
        type=float,
        required=False)
    parser.add_argument('-d', '--delimiter', help='Delimiter to use', type=str)
    parser.add_argument(
        '-k',
        '--kind',
        help='Kind of interpolation (linear -- default --, cubic, ...)',
        default='linear')
    parser.add_argument(
        '-f',
        '--fields',
        help='Fields (default: xy). Give the column field names (e.g. xyy)',
        default='xy')
    args = parser.parse_args()

    A = read_stdin(delimiter=args.delimiter)
    fields = parse_fields(args.fields)
    x, y = A[:, fields == 'x'], A[:, fields == 'y']
    if x.shape[1] == 1:
        x = np.squeeze(x)
    else:
        print('Multiple x value given. Not yet implemented')
        sys.exit()
    ys = []
    for y in y.T:
        xinter, yinter = interpolate(x,
                                     y,
                                     args.step,
                                     xmin=args.min,
                                     xmax=args.max,
                                     kind=args.kind)
        ys.append(yinter)
    ys = np.asarray(ys).T
    format_output(xinter, ys, args.delimiter)
