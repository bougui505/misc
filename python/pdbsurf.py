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
import os
import re
import numpy as np


def read_wrl(wrlfilename):
    """
    See: https://scicomp.stackexchange.com/a/12943/2121
    """
    data = []
    with open(wrlfilename, "r") as wrl:
        for lines in wrl:
            a = re.findall("[-0-9]{1,3}.[0-9]{6}", lines)
            if len(a) == 3:
                data.append(a)
    data = np.asarray(data, dtype=float)
    return data


def pdb_to_surf(pdbfilename, sel):
    cmd.reinitialize()
    cmd.load(pdbfilename, 'tosurf')
    selection = f'tosurf and {sel}'
    cmd.remove(f'not {sel}')
    coords = cmd.get_coords(selection)
    cmd.hide('everything')
    cmd.show_as('surface', selection)
    outwrl = f"{os.path.splitext(pdbfilename)[0]}.wrl"
    cmd.save(outwrl)
    pts = read_wrl(outwrl)
    pts += coords.mean(axis=0)
    return pts


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='Convert a pdb file to surface points')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-p', '--pdb')
    parser.add_argument('-s', '--sel', help='Selection ot atoms to compute the surface on (default: all)', default='all')
    args = parser.parse_args()

    pts = pdb_to_surf(args.pdb, args.sel)
    print(f"n_surface_pts: {pts.shape}")
    outbasename = f"{os.path.splitext(args.pdb)[0]}.surf"
    np.save(f'{outbasename}.npy', pts)
    np.savetxt(f'{outbasename}.xyz', np.c_[['n', ] * len(pts), np.asarray(pts, dtype=object)], fmt=['%s', '%.4f', '%.4f', '%.4f'])
