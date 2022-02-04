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

# Count the number of beta-strand in a beta sheet

import sys
from pymol import cmd
import numpy as np
import scipy.spatial.distance as scidist
import scipy.ndimage as ndi
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_coords(pdb, selection, traj=None):
    selection = f'myprot and {selection} and name CA'
    cmd.load(filename=pdb, object='myprot')
    natoms = cmd.select(selection)
    if traj is not None:
        cmd.load_traj(filename=traj,
                      object='myprot',
                      state=1,
                      selection=selection)
    coords = cmd.get_coords(selection=selection, state=0)
    nframes = coords.shape[0] // natoms
    coords = coords.reshape((nframes, natoms, 3))
    return coords


def get_contacts(coords, threshold=6):
    dmat = scidist.squareform(scidist.pdist(coords))
    cmap = dmat < threshold
    return cmap


def clean_contact(cmap):
    cmap = np.triu(cmap)
    return cmap


def get_label(cmap):
    labels, num_features = ndi.label(cmap)
    # Remove diagonal label
    label_diag = labels[0, 0]
    labels[labels == label_diag] = 0
    # Remove small clusters
    for label in np.unique(labels):
        count = (labels == label).sum()
        if count < 3:
            labels[labels == label] = 0
    return labels


def count_strands(coords, contact_threshold=6):
    # coords.shape must be (nframes, natoms, 3) with nframes=1 if a single conformation was loaded
    nstrands = []
    with tqdm(total=len(coords)) as pbar:
        for conf in coords:
            cmap = get_contacts(conf, threshold=contact_threshold)
            cmap = clean_contact(cmap)
            labels = get_label(cmap)
            # plt.matshow(labels)
            # plt.show()
            labelids = np.unique(labels[labels != 0])
            n = len(labelids)
            if n > 0:
                n += 1
            nstrands.append(n)
            pbar.update()
    return nstrands


def format_output(strandlist, outfilename=None, traj=None, selection=None):
    if outfilename is None:
        header = ''
        outfilename = sys.stdout
    else:
        header = f'Number of beta-strand for trajectory {traj} and selection {selection}'
    np.savetxt(outfilename, strandlist, fmt='%d', header=header)


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--pdb', required=True)
    parser.add_argument('--sel', required=True)
    parser.add_argument('--traj', required=False)
    parser.add_argument('--threshold',
                        help='Distance threshold in angstrom for contacts',
                        default=6.0,
                        type=float)
    parser.add_argument('--out', default='STRANDS.txt')
    args = parser.parse_args()

    coords = load_coords(pdb=args.pdb, traj=args.traj, selection=args.sel)
    nstrands = count_strands(coords, contact_threshold=args.threshold)
    format_output(nstrands,
                  traj=args.traj,
                  selection=args.sel,
                  outfilename=args.out)
