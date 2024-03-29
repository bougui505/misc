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

try:
    from pymol import cmd
except ImportError:
    print("pymol not imported some features will be missing")
    pass
import numpy as np
import sys
import os
from misc.protein.rotate import rotate as rotate_coords
from misc.protein.rotate import get_rotation_matrix
from misc.protein.rotate import rotate_pdb
import scipy.ndimage


def save_density(
    density, outfilename, spacing=1, origin=[0, 0, 0], padding=0, transpose=True
):
    """
    Save the density file as mrc for the given atomname
    """
    density = density.astype("float32")
    with mrcfile.new(outfilename, overwrite=True) as mrc:
        if transpose:
            density = density.T
        mrc.set_data(density)
        mrc.voxel_size = spacing
        mrc.header["origin"]["x"] = origin[0] - padding
        mrc.header["origin"]["y"] = origin[1] - padding
        mrc.header["origin"]["z"] = origin[2] - padding
        mrc.update_header_from_data()
        mrc.update_header_stats()


def mrc_to_array(mrcfilename, normalize=False, padding=0, return_dict=False):
    """
    Read a mrc file into a numpy array

    Args:
        mrcfilename:
        normalize:
        padding:

    Returns:
        If not return_dict:
            data
            origin
            spacing
        else:
            {'grid': data, 'origin': origin, 'spacing': spacing}

    """
    with mrcfile.open(mrcfilename) as mrc:
        x0 = mrc.header["origin"]["x"]
        y0 = mrc.header["origin"]["y"]
        z0 = mrc.header["origin"]["z"]
        origin = np.asarray([x0, y0, z0])
        spacing = mrc.voxel_size
        data = mrc.data.copy()
        if normalize:
            data -= data.min()
            data /= data.max()
        if padding > 0:
            data = np.pad(data, pad_width=padding)
            origin -= padding
        spacing = str(spacing).replace("(", "").replace(")", "").split(",")
        spacing = np.asarray(spacing, dtype=float)
        assert (spacing == spacing[0]).all()
        spacing = spacing[0]
        if not return_dict:
            return data, origin, spacing
        else:
            return {"grid": data, "origin": origin, "spacing": spacing}


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


def digitize(coords, data, origin, spacing):
    n, p, q = data.shape
    xbins = np.arange(start=origin[0], stop=n * spacing, step=spacing)
    ybins = np.arange(start=origin[1], stop=p * spacing, step=spacing)
    zbins = np.arange(start=origin[2], stop=q * spacing, step=spacing)
    coords = np.atleast_2d(coords)
    i = np.digitize(coords[:, 0], xbins)
    j = np.digitize(coords[:, 1], ybins)
    k = np.digitize(coords[:, 2], zbins)
    inds = np.asarray([i, j, k])
    inds = np.squeeze(inds)
    return inds


def mrc_to_pdb(
    mrcfilename, outpdb, minthr=-np.inf, maxthr=np.inf, stride=1, normalize=False
):
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
        sys.stdout.write(f"Saving grid-point: {resi+1}/{n}          \r")
        sys.stdout.flush()
        cmd.pseudoatom(
            pos=(x, y, z),
            object="out",
            state=1,
            resi=resi + 1,
            chain="Z",
            name="H",
            elem="H",
            b=distrib[resi],
        )
    cmd.save(outpdb, "out")


def translate_pdb(pdbfilename, outpdbfilename, translation):
    cmd.reinitialize()
    cmd.load(pdbfilename, object="inpdb")
    coords = cmd.get_coords(selection="inpdb")
    coords += translation
    cmd.load_coords(coords, selection="inpdb")
    cmd.save(outpdbfilename, selection="inpdb")


def rotate(grid, rotation_matrix=None, angle_x=0, angle_y=0, angle_z=0, center=None):
    n, p, q = grid.shape
    outgrid = np.zeros_like(grid)
    counts = np.zeros_like(grid)
    coords, values = filter_by_condition(grid, np.ones_like(grid, dtype=np.bool))
    coords = np.float_(coords)
    if rotation_matrix is None:
        rotation_matrix = get_rotation_matrix(angle_x, angle_y, angle_z)
    coords_new = rotate_coords(coords, rotation_matrix=rotation_matrix, center=center)
    coords_new = np.int_(np.round(coords_new))
    sel = np.all(coords_new > 0, axis=1)
    sel = sel & (coords_new[:, 0] < n)
    sel = sel & (coords_new[:, 1] < p)
    sel = sel & (coords_new[:, 2] < q)
    coords_new = coords_new[sel]
    values = values[sel]
    np.add.at(counts, tuple(coords_new.T), 1.0)
    np.add.at(outgrid, tuple(coords_new.T), values)
    outgrid = np.divide(
        outgrid, counts, where=(counts != 0), out=np.zeros_like(outgrid)
    )
    return outgrid


def resample(grid, spacing, zoom):
    """
    >>> grid = np.random.uniform(low=0., high=1., size=(10, 20, 30))
    >>> spacing = 1.
    >>> out, spacing_out = resample(grid, spacing, 2)
    >>> out.shape
    (20, 40, 60)
    >>> spacing_out
    0.5
    """
    out = scipy.ndimage.zoom(grid, zoom=zoom)
    spacing_out = spacing / zoom
    return out, spacing_out


if __name__ == "__main__":
    import doctest
    import argparse

    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument(
        "--npy", help="Read a npy file and create a MRC file (see: --out)"
    )
    parser.add_argument(
        "--origin",
        type=float,
        nargs="+",
        default=[0.0, 0.0, 0.0],
        help="Origin for the MRC file",
    )  # 10. 20. 30.
    parser.add_argument("--center", help="Set the origin to 0 0 0", action="store_true")
    parser.add_argument(
        "--resample",
        help="Resample the grid by the given ratio. >1: bigger grid. <1: smaller grid",
        type=float,
    )
    parser.add_argument(
        "--rotate", help="x, y, z angles of rotation in degree", nargs="+", type=float
    )
    parser.add_argument(
        "--pdb",
        help="PDB file to translate when center option is called in order to align the pdb and the centered grid",
    )
    parser.add_argument(
        "--spacing", type=float, default=1, help="Spacing for the MRC file"
    )  # 1.
    parser.add_argument("--out", help="Out MRC file name (see: --npy)")  # 10. 20. 30.
    parser.add_argument(
        "--mrc",
        help="Read the given MRC and print the flatten data to stdout if --outpdb not given",
    )
    parser.add_argument("--outpdb", help="Convert the given mrc file (--mrc) to pdb")
    parser.add_argument(
        "--outnpy", help="Convert the given mrc file (--mrc) to npy file"
    )
    parser.add_argument(
        "--minthr",
        help="Minimum threshold the MRC to save to pdb (--outpdb)",
        default=-np.inf,
        type=float,
    )
    parser.add_argument(
        "--maxthr",
        help="Maximum threshold the MRC to save to pdb (--outpdb)",
        default=np.inf,
        type=float,
    )
    parser.add_argument(
        "--stride",
        help="Stride for the grid to save to pdb (--outpdb), default=1",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--normalize", help="Normalize the density between 0 and 1", action="store_true"
    )
    parser.add_argument(
        "--info",
        help="Print informations about the given mrc (see --mrc)",
        action="store_true",
    )
    parser.add_argument(
        "--padding", help="Add a padding to the given map", type=int, default=0
    )
    parser.add_argument("--test", help="Test the code", action="store_true")
    args = parser.parse_args()

    if args.test:
        doctest.testmod(
            optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE
        )
        sys.exit()

    if args.pdb is not None:
        cmd.load(args.pdb, object="pdbin")
    if args.npy is not None:
        data = np.load(args.npy)
        transpose = True
    if args.mrc is not None:
        data, origin_in, args.spacing = mrc_to_array(
            args.mrc, normalize=args.normalize, padding=args.padding
        )
        if args.resample is not None:
            data, args.spacing = resample(data, args.spacing, args.resample)
        if args.center:
            args.origin = np.asarray([0, 0, 0])
            translation = -origin_in
            if args.pdb is not None:
                outpdbname = f"{os.path.splitext(args.out)[0]}.pdb"
                translate_pdb(args.pdb, outpdbname, translation)
        if args.rotate is not None:
            alpha, beta, gamma = np.deg2rad(args.rotate)
            data = rotate(data, angle_x=-gamma, angle_y=beta, angle_z=-alpha)
            if args.pdb is not None:
                outpdbname = f"{os.path.splitext(args.out)[0]}.pdb"
                rotate_pdb(
                    args.pdb, outpdbname, angle_x=alpha, angle_y=beta, angle_z=gamma
                )
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
    if (
        args.outpdb is None
        and args.outnpy is None
        and args.out is None
        and not args.info
    ):
        np.savetxt(sys.stdout, data.flatten())
    if args.outpdb is not None:
        mrc_to_pdb(
            args.mrc,
            args.outpdb,
            minthr=args.minthr,
            maxthr=args.maxthr,
            stride=args.stride,
            normalize=args.normalize,
        )
    if args.outnpy is not None:
        np.save(args.outnpy, data)
    if args.out is not None:
        save_density(data, args.out, args.spacing, args.origin, 0, transpose=transpose)
