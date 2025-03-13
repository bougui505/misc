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
from pymol import cmd
from scipy.spatial.transform import Rotation as R


def orient(coords, flip=False):
    """
    >>> import misc.protein.coords_loader as coords_loader
    >>> coords = coords_loader.get_coords(pdb='1ycr')
    Fetching 1ycr from the PDB
    >>> coords
    array([[ 10.801, -12.147,  -5.18 ],
           [ 11.124, -13.382,  -4.414],
           [ 11.769, -12.878,  -3.13 ],
           ...,
           [ 14.536, -16.773,  -4.071],
           [ 12.963, -18.387,  -4.023],
           [ 16.296, -17.351,   0.139]], dtype=float32)
    >>> coords = orient(coords)
    >>> coords
    array([[-14.8789215,  10.66122  ,  -4.8155413],
           [-13.844462 ,  11.316589 ,  -3.9689634],
           [-14.034291 ,  10.702146 ,  -2.588702 ],
           ...,
           [ -9.175905 ,  10.936841 ,  -2.8206384],
           [ -8.906612 ,  13.1606245,  -3.0734174],
           [ -8.294708 ,  10.60735  ,   1.6816623]], dtype=float32)
    """
    coords -= coords.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(coords.T.dot(coords))
    print(f"{eigenvectors}")
    if flip:
        print("flipping")
        eigenvectors[:, 0] *= -1
    print(f"{eigenvectors}")
    det = np.linalg.det(eigenvectors)
    if det < 0:
        print("regularizing")
        eigenvectors[:, -1] *= -1
    print(f"{eigenvectors}")
    coords = coords.dot(eigenvectors)
    return coords


def get_rotation_matrix(angle_x, angle_y, angle_z):
    axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    rotation_matrix = np.identity(3)
    for angle, vec in zip([angle_x, angle_y, angle_z], axes):
        rotation_matrix = rotation_matrix.dot(R.from_rotvec(angle * vec).as_matrix())
    return rotation_matrix


def rotate(coords, rotation_matrix=None, angle_x=0, angle_y=0, angle_z=0, center=None):
    """
    Rotate 3D coordinates given a rotation matrix or angles for each axis

    Args:
        coords:
        rotation_matrix:
        angle_x: angle around x-axis in radian
        angle_y: angle around y-axis in radian
        angle_z: angle around z-axis in radian
        center: center of the rotation. If None, the geometric center is defined as the origin

    Returns:
        out_coords: The rotated 3D coordinates

    """
    if center is None:
        center = coords.mean(axis=0)
    coords -= center
    if rotation_matrix is None:
        rotation_matrix = get_rotation_matrix(angle_x, angle_y, angle_z)
    out_coords = (rotation_matrix.dot(coords.T)).T + center
    return out_coords


def rotate_pdb(pdbin, pdbout, rotation_matrix=None, angle_x=0, angle_y=0, angle_z=0, trans=np.zeros(3)):
    cmd.reinitialize()
    cmd.load(pdbin, object='inpdb')
    coords = cmd.get_coords()
    coords_rot = rotate(coords, angle_x=angle_x, angle_y=angle_y, angle_z=angle_z)
    coords_rot += np.atleast_2d(trans)
    cmd.load_coords(coords_rot, 'inpdb')
    cmd.save(pdbout)


if __name__ == '__main__':
    import argparse

    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--pdb', help='pdb file to rotate')
    parser.add_argument('--orient', action='store_true')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--out', help='out pdb filename')
    parser.add_argument('--alpha', help='Angle (deg.) for x axis', type=float, default=0.)
    parser.add_argument('--beta', help='Angle (deg.) for y axis', type=float, default=0.)
    parser.add_argument('--gamma', help='Angle (deg.) for z axis', type=float, default=0.)
    args = parser.parse_args()

    if args.orient:
        cmd.reinitialize()
        cmd.load(args.pdb, object='inpdb')
        coords = cmd.get_coords()
        coords_o = orient(coords, flip=args.flip)
        print(f"{coords_o.mean(axis=0)=}")  # type: ignore
        cmd.load_coords(coords_o, 'inpdb')
        cmd.save(args.out)

    if args.alpha or args.beta or args.gamma:
        alpha = np.deg2rad(args.alpha)
        beta = np.deg2rad(args.beta)
        gamma = np.deg2rad(args.gamma)
        rotate_pdb(args.pdb, args.out, angle_x=alpha, angle_y=beta, angle_z=gamma)
