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

import os
import numpy as np
from pymol import cmd
from tqdm import tqdm


def get_nss(pdb, ssclass, sel='all', traj=None, outfile=None, stride=1):
    """
    Get the number of atoms with the given SS-class ('H' or 'S')
    """
    cmd.load(filename=pdb, object='myprot')
    if outfile is not None:
        if not os.path.exists(outfile):
            outfile = open(outfile, 'w')
            outfile.write(f'#state #n({ssclass})\n')
            laststate = 0
        else:
            data = np.genfromtxt(outfile, dtype=int)
            laststate = data[-1][0]
            outfile = open(outfile, 'a')
    if traj is not None:
        # improve PyMOL performance for many-state objects
        cmd.set('defer_builds_mode', 3)
        cmd.load_traj(traj,
                      object='myprot',
                      state=1,
                      selection=sel,
                      start=laststate + stride)
    nstates = cmd.count_states(selection='myprot')
    print(f'Computing secondary structures for {nstates} states')
    if nstates > 1:
        pbar = tqdm(total=nstates)  # Init pbar
    for state in range(1, nstates + 1, stride):
        cmd.dss(selection='all', state=state)
        nss = cmd.select(
            f'ss {ssclass} and {sel} and state {state} and name CA')
        if outfile is None or nstates == 1:
            print(nss)
        else:
            outfile.write(f'{laststate + state} {nss}\n')
            pbar.update(stride)
    if outfile is not None:
        outfile.close()
    if nstates > 1:
        pbar.close()


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-p', '--pdb', required=True)
    parser.add_argument('-t', '--traj', required=False)
    parser.add_argument('-s',
                        '--ss',
                        help='SS class to detect (H or S)',
                        required=True)
    parser.add_argument('--sel',
                        help='Selection to compute the SS analysis on',
                        default='all')
    parser.add_argument(
        '--out',
        help='Outfilename to store the result. If not given print to stdout')
    parser.add_argument(
        '--stride',
        help='Stride for the secondary structure computation (default: 1)',
        type=int,
        default=1)
    args = parser.parse_args()

    get_nss(pdb=args.pdb,
            ssclass=args.ss,
            traj=args.traj,
            sel=args.sel,
            outfile=args.out,
            stride=args.stride)
