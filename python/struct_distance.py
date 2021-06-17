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

from pymol import cmd
import numpy as np

if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='Compute distance between two selections. If the selection has multiple atoms compute the geometric center and then the distance between the geometric centers')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--top', help='Topology for the trajectory', required=True)
    parser.add_argument('--traj', help='Trajectory file, loaded as traj', required=True)
    parser.add_argument('--ref', help='Reference structure, loaded as ref', required=True)
    parser.add_argument('--sel1', help='First selection', required=True)
    parser.add_argument('--sel2', help='Second selection', required=True)
    parser.add_argument('-o', '--out', help='Out file name to store the distances', required=True)
    # parser.add_argument('--align', help='Align the reference on the trajectory (default: do not align)', action='store_true')
    args = parser.parse_args()
    
    cmd.load(args.top, 'traj')
    cmd.load_traj(args.traj, 'traj', state=1)
    nstates = cmd.count_states('traj')
    print(f'traj_nframes: {nstates}')
    cmd.load(args.ref, 'ref')
    coords1 = cmd.get_coords(selection=args.sel1, state=0)
    nstates1 = cmd.count_states(args.sel1)
    natoms1 = cmd.select(args.sel1, state=1)
    if nstates1 > 1:
        coords1 = coords1.reshape((nstates1, natoms1, 3))
    coords2 = cmd.get_coords(selection=args.sel2, state=0)
    nstates2 = cmd.count_states(args.sel2)
    natoms2 = cmd.select(args.sel2, state=1)
    if nstates2 > 1:
        coords2 = coords2.reshape((nstates2, natoms2, 3))
    nstates2 = cmd.count_states(args.sel2)
    print(f'sel1_shape: {coords1.shape}')
    print(f'sel2_shape: {coords2.shape}')
    gc1 = coords1.mean(axis=-2)
    gc2 = coords2.mean(axis=-2)
    print(f'geometric_center_1: {gc1.shape}')
    print(f'geometric_center_2: {gc2.shape}')
    distances = np.linalg.norm(gc2 - gc1, axis=-1)
    np.savetxt(args.out, distances, fmt='%.4f')
