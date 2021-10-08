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
from scipy.spatial.transform import Rotation as R


def get_rotation_matrix(angle_x, angle_y, angle_z):
    axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    rotation_matrix = np.identity(3)
    for angle, vec in zip([angle_x, angle_y, angle_z], axes):
        rotation_matrix = rotation_matrix.dot(
            R.from_rotvec(angle * vec).as_matrix())
    return rotation_matrix


def rotate(coords, rotation_matrix=None, angle_x=0, angle_y=0, angle_z=0):
    """
    Rotate 3D coordinates given a rotation matrix or angles for each axis

    Args:
        coords:
        rotation_matrix:
        angle_x:
        angle_y:
        angle_z:

    Returns:
        out_coords: The rotated 3D coordinates

    """
    center = coords.mean(axis=0)
    coords -= center
    if rotation_matrix is None:
        rotation_matrix = get_rotation_matrix(angle_x, angle_y, angle_z)
    out_coords = (rotation_matrix.dot(coords.T)).T + center
    return out_coords


if __name__ == '__main__':
    from pymol import cmd
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--pdb', help='pdb file to rotate', required=True)
    parser.add_argument('--out', help='out pdb filename', required=True)
    parser.add_argument('--alpha',
                        help='Angle (deg.) for x axis',
                        type=float,
                        default=0.)
    parser.add_argument('--beta',
                        help='Angle (deg.) for y axis',
                        type=float,
                        default=0.)
    parser.add_argument('--gamma',
                        help='Angle (deg.) for z axis',
                        type=float,
                        default=0.)
    args = parser.parse_args()

    cmd.load(args.pdb, object='inpdb')
    coords = coords = cmd.get_coords()
    alpha = np.deg2rad(args.alpha)
    beta = np.deg2rad(args.beta)
    gamma = np.deg2rad(args.gamma)
    coords_rot = rotate(coords, angle_x=alpha, angle_y=beta, angle_z=gamma)
    cmd.load_coords(coords_rot, 'inpdb')
    cmd.save(args.out)
