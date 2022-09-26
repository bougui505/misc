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

TIMER = Timer(autoreset=True)


def gaussian(x, y, z, x0, y0, z0, sigma):
    return np.exp(-((x - x0)**2 / (2 * sigma**2) + (y - y0)**2 / (2 * sigma**2) + (z - z0)**2 / (2 * sigma**2)))


def gaussians(x, y, z, center_list, sigma):
    """
    """
    out = 0
    neighbors = neighbor_lists(center_list, np.c_[x, y, z], radius=3. * sigma)
    assert len(neighbors) == len(center_list)
    for i, center in enumerate(center_list):
        x0, y0, z0 = center
        density_values = np.zeros(len(x))
        density_values[neighbors[i]] = gaussian(x[neighbors[i]], y[neighbors[i]], z[neighbors[i]], x0, y0, z0, sigma)
        out += density_values
    return out


def neighbor_lists(coords1, coords2, radius):
    """
    >>> coords = get_coords('1ycr', verbose=False)
    >>> GX, GY, GZ, origin = Grid(coords, padding=3, spacing=1)
    >>> grid_coords = np.c_[GX.flatten(), GY.flatten(), GZ.flatten()]
    >>> coords.shape
    (818, 3)
    >>> grid_coords.shape
    (55104, 3)
    >>> neighbors = neighbor_lists(coords, grid_coords, radius=4.5)
    """
    tree = KDTree(coords2)
    neighbors = tree.query_radius(coords1, radius)
    return neighbors


def Gaussians(pdb, sigma, selection='all', verbose=False):
    """
    >>> gaussians = Gaussians('1ycr', sigma=3.)
    >>> gaussians
    functools.partial(<function gaussians at 0x...

    # >>> gaussians(29.08331, -16.980543, -4.387978)
    # 18.031352571395498

    >>> coords = get_coords('1ycr', verbose=False)
    >>> gaussians(*coords.T).shape
    (818,)
    """
    coords = get_coords(pdb, selection=selection, verbose=verbose)
    if verbose:
        TIMER.start('Computing Gaussians')
    if verbose:
        TIMER.stop()
    return functools.partial(gaussians, center_list=coords, sigma=sigma)


def Grid(coords, padding, spacing):
    x_min, y_min, z_min = coords.min(axis=0) - np.asarray(padding)
    x_max, y_max, z_max = coords.max(axis=0) + np.asarray(padding)
    origin = (x_min, y_min, z_min)
    X = np.arange(x_min, x_max, spacing)
    Y = np.arange(y_min, y_max, spacing)
    Z = np.arange(z_min, z_max, spacing)
    GX, GY, GZ = np.meshgrid(X, Y, Z, indexing='ij')
    return GX, GY, GZ, origin


def Density(pdb, sigma, spacing, padding=(0, 0, 0), selection='all', verbose=False):
    """
    >>> density, origin = Density('1ycr', sigma=3, spacing=1)
    >>> density.shape
    (35, 36, 26)
    >>> density, origin = Density('1ycr', sigma=3, spacing=1, padding=3)
    >>> density.shape
    (41, 42, 32)
    """
    coords = get_coords(pdb, selection=selection, verbose=verbose)
    GX, GY, GZ, origin = Grid(coords, padding, spacing)
    nX, nY, nZ = GX.shape
    gaussians = Gaussians(pdb=pdb, sigma=sigma, selection=selection, verbose=verbose)
    density = []
    if verbose:
        TIMER.start('Adding gaussians')
    density = gaussians(GX.flatten(), GY.flatten(), GZ.flatten()).reshape((nX, nY, nZ))
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
    parser.add_argument('--sigma', type=float, default=1.5)
    parser.add_argument('--spacing', type=int, default=1)
    parser.add_argument('--padding', type=int, default=3)
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()

    density, origin = Density(args.pdb, sigma=args.sigma, spacing=args.spacing, padding=args.padding, verbose=True)
    outfilename = f'{os.path.splitext(args.pdb)[0]}_density.mrc'
    mrc.save_density(density, outfilename, spacing=args.spacing, padding=0, origin=origin)
