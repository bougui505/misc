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

import pymol
import os
from pymol import cmd
from psico import fullinit

cmd.set('fetch_path', os.path.expanduser('~/pdb'))

if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-p',
                        '--pdbs',
                        help='List of PDBs to process',
                        nargs='+',
                        required=True)
    parser.add_argument('-s',
                        '--select',
                        help='Selection for the RMSD calculation',
                        default='all')
    parser.add_argument('--tmscore',
                        help='Compute the TM-Score instead of the RMSD',
                        action='store_true')
    args = parser.parse_args()

    npdbs = len(args.pdbs)

    for i, pdb in enumerate(args.pdbs):
        try:
            cmd.load(pdb, f'pdb{i}')
        except pymol.CmdException:
            cmd.fetch(code=pdb, name=f'pdb{i}')

    for i in range(npdbs - 1):
        for j in range(i + 1, npdbs):
            if args.tmscore:
                tmscore = cmd.tmalign(f'pdb{i} and {args.select}',
                                      f'pdb{j} and {args.select}',
                                      quiet=1)
            else:
                rmsd = cmd.align(f'pdb{i} and {args.select}',
                                 f'pdb{j} and {args.select}')[0]
            print(f'pdb1: {args.pdbs[i]}')
            print(f'pdb2: {args.pdbs[j]}')
            if args.tmscore:
                print(f'tmscore: {tmscore:.4f}')
            else:
                print(f'rmsd: {rmsd:.4f}')
            print()
