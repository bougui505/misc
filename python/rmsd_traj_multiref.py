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
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-p', '--pdb')
    parser.add_argument('-t', '--traj')
    parser.add_argument('-s', '--sel', default='all')
    parser.add_argument('-r', '--refs', nargs='+')
    parser.add_argument('-o', '--out', default='rmsds.txt')
    args = parser.parse_args()

    cmd.load(args.pdb, 'trajin')
    cmd.load_traj(args.traj, 'trajin', state=1)
    nframes = cmd.count_states('trajin')
    print(f'Number of frames: {nframes}')
    for ref in args.refs:
        print(f'Loading {ref}')
        cmd.load(ref, 'trajin')
    cmd.remove('hydro and trajin')
    nref = len(args.refs)
    rmsds_all = []
    for ind, frameref in enumerate([nframes + i for i in range(1, nref + 1)]):
        print(f'Computing RMSD for {args.refs[ind]}')
        rmsds = cmd.intra_fit(f'trajin and {args.sel}', state=frameref)[:nframes]
        rmsds_all.append(rmsds)
    rmsds_all = np.asarray(rmsds_all, dtype=object).T
    min_rmsd = rmsds_all.min(axis=1)
    args_min = rmsds_all.argmin(axis=1)
    args_min = np.asarray(args.refs)[args_min]
    np.savetxt(args.out, np.c_[rmsds_all, min_rmsd, args_min], fmt=' '.join(['%.4f', ] * (nref + 1)) + ' %s', header=' '.join([f'#{ref}' for ref in args.refs]) + ' #min_rmsd' + ' #min_rmsd_ref', comments='')
