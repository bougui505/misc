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

import pandas as pd
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{script_dir}/../../python')
import recutils


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-r', '--rec', help='Rec file with TM-score and RMSD results as given by tmscore-multi', required=True)
    parser.add_argument('--rmsd', help='Plot the RMSD instead of the TM-score', action='store_true')
    parser.add_argument('--annot', help='Annotate the plot with the values', action='store_true')
    args = parser.parse_args()

    rec = recutils.load(args.rec)
    rec = pd.DataFrame(rec)
    if args.rmsd:
        pmat = rec.pivot(index='native', columns='model', values='rmsd')
        cmap_label = 'RMSD (Å)'
        Z = hierarchy.ward(pmat.values)
    else:
        pmat = rec.pivot(index='native', columns='model', values='tmscore')
        cmap_label = 'TM-score'
        Z = hierarchy.ward(1. - pmat.values)
    pmat.rename(columns={e: os.path.splitext(os.path.basename(e))[0] for e in pmat.columns.values}, inplace=True)
    pmat.rename(index={e: os.path.splitext(os.path.basename(e))[0] for e in pmat.index.values}, inplace=True)
    order = hierarchy.leaves_list(Z)
    order = list(pmat.columns.values[order])
    pmat = pmat.reindex(order)[order]
    sns.heatmap(pmat, annot=args.annot, cbar_kws={'label': cmap_label})
    plt.show()
