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
from misc.protein.coords_loader import get_coords
import numpy as np
from misc.Grid3 import mrc
import functools
from misc.Timer import Timer
from sklearn.neighbors import KDTree
import tqdm
import scipy.ndimage

TIMER = Timer(autoreset=True)


def Grid(coords, padding, spacing, return_axis=False):
    x_min, y_min, z_min = coords.min(axis=0) - np.asarray(padding)
    x_max, y_max, z_max = coords.max(axis=0) + np.asarray(padding)
    origin = (x_min, y_min, z_min)
    X = np.arange(x_min, x_max, spacing)
    Y = np.arange(y_min, y_max, spacing)
    Z = np.arange(z_min, z_max, spacing)
    if return_axis:
        return X, Y, Z, origin
    GX, GY, GZ = np.meshgrid(X, Y, Z, indexing='ij')
    return GX, GY, GZ, origin


def Density(pdb,
            sigma,
            spacing,
            padding=(0, 0, 0),
            selection='all',
            rotation=None,
            random_rotation=False,
            random_chains=False,
            verbose=False,
            obj=None):
    """
    rotation: (angle_x, angle_y, angle_z) -- in radian -- for coordinates rotation. Default: no rotation

    >>> density, origin = Density('1ycr', sigma=3, spacing=1)
    >>> density.shape
    (34, 35, 25)
    >>> density, origin = Density('1ycr', sigma=3, spacing=1, padding=3)
    >>> density.shape
    (40, 41, 31)
    """
    coords = get_coords(pdb,
                        selection=selection,
                        verbose=verbose,
                        random_rotation=random_rotation,
                        split_by_chains=random_chains,
                        obj=obj)
    if random_chains:
        nchains = len(coords)
        chainids = np.random.choice(nchains, size=np.random.choice(nchains) + 1, replace=False)
        if verbose:
            print(f'Number of chains: {nchains}')
            print(f'Selected chains: {chainids}')
        coords = np.concatenate([coords[i] for i in chainids], axis=0)
    if verbose:
        TIMER.start('Computing density')
    X, Y, Z, origin = Grid(coords, padding, spacing, return_axis=True)
    density, edges = np.histogramdd(coords, bins=(X, Y, Z))
    origin = (edges[0][0] + spacing / 2, edges[1][0] + spacing / 2, edges[2][0] + spacing / 2)
    density = scipy.ndimage.gaussian_filter(density, sigma)
    if verbose:
        TIMER.stop()
    return density, origin


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir


if __name__ == '__main__':
    import sys
    import doctest
    import argparse
    # ### UNCOMMENT FOR LOGGING ####
    # import os
    # import logging
    # logfilename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    # logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    # logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-p', '--pdb')
    parser.add_argument('--selection', default='all')
    parser.add_argument('--sigma', type=float, default=1.5)
    parser.add_argument('--spacing', type=float, default=1)
    parser.add_argument('--padding', type=int, default=3)
    parser.add_argument('--random_rotation', action='store_true')
    parser.add_argument('--random_chains', action='store_true')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()

    density, origin = Density(args.pdb,
                              sigma=args.sigma,
                              spacing=args.spacing,
                              padding=args.padding,
                              verbose=True,
                              selection=args.selection,
                              random_rotation=args.random_rotation,
                              random_chains=args.random_chains)
    outfilename = f'{os.path.splitext(args.pdb)[0]}_density.mrc'
    mrc.save_density(density, outfilename, spacing=args.spacing, padding=0, origin=origin)
