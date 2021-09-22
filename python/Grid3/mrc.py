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

import mrcfile
from pymol import cmd
import numpy as np
import sys


def save_density(density,
                 outfilename,
                 spacing=1,
                 origin=[0, 0, 0],
                 padding=0,
                 transpose=True):
    """
    Save the density file as mrc for the given atomname
    """
    density = density.astype('float32')
    with mrcfile.new(outfilename, overwrite=True) as mrc:
        if transpose:
            density = density.T
        mrc.set_data(density)
        mrc.voxel_size = spacing
        mrc.header['origin']['x'] = origin[0] - padding
        mrc.header['origin']['y'] = origin[1] - padding
        mrc.header['origin']['z'] = origin[2] - padding
        mrc.update_header_from_data()
        mrc.update_header_stats()


def mrc_to_array(mrcfilename, normalize=False, padding=0):
    """
    Print the MRC values on stdout
    """
    with mrcfile.open(mrcfilename) as mrc:
        x0 = mrc.header['origin']['x']
        y0 = mrc.header['origin']['y']
        z0 = mrc.header['origin']['z']
        origin = np.asarray([x0, y0, z0])
        spacing = mrc.voxel_size
        data = mrc.data.copy()
        if normalize:
            data -= data.min()
            data /= data.max()
        if padding > 0:
            data = np.pad(data, pad_width=padding)
            origin -= padding
        spacing = str(spacing).replace('(', '').replace(')', '').split(',')
        spacing = np.asarray(spacing, dtype=float)
        assert (spacing == spacing[0]).all()
        spacing = spacing[0]
        return data, origin, spacing


def filter_by_condition(grid, condition):
    """
    takes a grid and returns the coordinates of the non_zero elements corresponding to a condition
     as well as the associated values
    :param grid:
    :param condition: something like grid >0 : a numpy array of booleans and same size that grid
    :return:
    """
    zeroed = grid * condition
    coords = np.argwhere(zeroed)
    distrib = grid[condition]
    # Additional filtering
    distrib = distrib[distrib > 0]
    return coords, distrib


def mrc_to_pdb(mrcfilename,
               outpdb,
               minthr=-np.inf,
               maxthr=np.inf,
               stride=1,
               normalize=False):
    """
    Create a pdb file from the given mrcfilename
    """
    grid, origin, spacing = mrc_to_array(mrcfilename, normalize=normalize)
    grid = grid[::stride, ::stride, ::stride].T
    selection = np.logical_and(grid > minthr, grid <= maxthr)
    coords, distrib = filter_by_condition(grid, selection)
    n = coords.shape[0]
    coords = coords * stride
    coords = coords + origin
    for resi, (x, y, z) in enumerate(coords):
        sys.stdout.write(f'Saving grid-point: {resi+1}/{n}          \r')
        sys.stdout.flush()
        cmd.pseudoatom(pos=(x, y, z),
                       object='out',
                       state=1,
                       resi=resi + 1,
                       chain="Z",
                       name="H",
                       elem="H",
                       b=distrib[resi])
    cmd.save(outpdb, 'out')


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument(
        '--npy', help='Read a npy file and create a MRC file (see: --out)')
    parser.add_argument('--origin',
                        type=float,
                        nargs='+',
                        default=None,
                        help='Origin for the MRC file')  # 10. 20. 30.
    parser.add_argument('--spacing',
                        type=float,
                        default=1,
                        help='Spacing for the MRC file')  # 1.
    parser.add_argument('--out',
                        help='Out MRC file name (see: --npy)')  # 10. 20. 30.
    parser.add_argument(
        '--mrc',
        help=
        'Read the given MRC and print the flatten data to stdout if --outpdb not given'
    )
    parser.add_argument('--outpdb',
                        help='Convert the given mrc file (--mrc) to pdb')
    parser.add_argument('--outnpy',
                        help='Convert the given mrc file (--mrc) to npy file')
    parser.add_argument(
        '--minthr',
        help='Minimum threshold the MRC to save to pdb (--outpdb)',
        default=-np.inf,
        type=float)
    parser.add_argument(
        '--maxthr',
        help='Maximum threshold the MRC to save to pdb (--outpdb)',
        default=np.inf,
        type=float)
    parser.add_argument(
        '--stride',
        help='Stride for the grid to save to pdb (--outpdb), default=1',
        default=1,
        type=int)
    parser.add_argument('--normalize',
                        help='Normalize the density between 0 and 1',
                        action='store_true')
    parser.add_argument(
        '--info',
        help='Print informations about the given mrc (see --mrc)',
        action='store_true')
    parser.add_argument('--padding',
                        help='Add a padding to the given map',
                        type=int,
                        default=0)
    args = parser.parse_args()

    if args.npy is not None:
        data = np.load(args.npy)
        transpose = True
    if args.mrc is not None:
        data, origin_in, args.spacing = mrc_to_array(args.mrc,
                                                     normalize=args.normalize,
                                                     padding=args.padding)
        if args.origin is None:
            args.origin = origin_in
        transpose = False
    if args.info:
        nx, ny, nz = data.shape
        print(f"shape: {data.shape}")
        print(f"origin: {args.origin}")
        print(f"spacing: {args.spacing}")
        print(f"min_density: {data.min():.6g}")
        print(f"max_density: {data.max():.6g}")
        print(f"mean_density: {data.mean():.6g}")
    if args.outpdb is None and args.outnpy is None and args.out is None and not args.info:
        np.savetxt(sys.stdout, data.flatten())
    if args.outpdb is not None:
        mrc_to_pdb(args.mrc,
                   args.outpdb,
                   minthr=args.minthr,
                   maxthr=args.maxthr,
                   stride=args.stride,
                   normalize=args.normalize)
    if args.outnpy is not None:
        np.save(args.outnpy, data)
    if args.out is not None:
        save_density(data,
                     args.out,
                     args.spacing,
                     args.origin,
                     0,
                     transpose=transpose)
