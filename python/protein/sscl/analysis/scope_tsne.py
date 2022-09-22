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
import os
from scopeit import load_index, read_homologs
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from umap import UMAP
import numpy as np
import matplotlib.pyplot as plt
from misc.Timer import Timer


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir


def get_mapping_scope40(scopfile='../data/dir.des.scope.2.08-stable.txt'):
    """
    Get a mapping from filename to the chain they represent
    @param index_file:
    @return:

    >>> get_mapping_scope40()
    {'1ux8_A': 'a.1.1.1', '1dlw_A': 'a.1.1.1', '1uvy_A': 'a.1.1.1', '1dly_A': 'a.1.1.1', ...
    """
    mapping = {}
    with open(scopfile, 'r') as index_file:
        for line in index_file:
            splitted_line = line.split()
            if len(splitted_line) > 3 and not splitted_line[0] == "#" and not splitted_line[3] == '-':
                pdb, chain = splitted_line[4], splitted_line[5].split(':')[0]
                scope_class = splitted_line[2]
                mapping[f'{pdb}_{chain}'] = scope_class
    return mapping


def embed_scope():
    """
    >>> embed_scope()
    """
    mapping = get_mapping_scope40()
    findex, ids = load_index('../index.faiss')
    homologs = read_homologs('../data/homologs_scope40.txt.gz')
    scope40 = [e[0] for e in homologs]
    vlatents = []
    scopeclasses = []
    for pdbid in scope40:
        fid = ids[str(pdbid)]
        vlatents.append(findex.reconstruct(fid))
        scopeclasses.append(mapping[str(pdbid)])
    scopeclasses = np.asarray(scopeclasses)
    vlatents = np.asarray(vlatents)
    return vlatents, scopeclasses


def plot_latent(vlatents, scopeclasses, subsample=None, level=4, filtered_labels=['l']):
    """
    filtered_labels: labels to remove from the analysis (l is the Artifacts class)
    """
    # print(vlatents.shape, vlatents.dtype)
    if level < 4:
        scopeclasses = np.asarray(['.'.join(e.split('.')[:level]) for e in scopeclasses])
        print(scopeclasses)
    if subsample is not None:
        n = len(vlatents)
        choice = np.random.choice(n, size=subsample)
        vlatents = vlatents[choice]
        scopeclasses = scopeclasses[choice]
    if filtered_labels is not None:
        sel = ~np.isin(scopeclasses, filtered_labels)
        scopeclasses = scopeclasses[sel]
        vlatents = vlatents[sel]
    timer = Timer(autoreset=True)
    # print('PCA ...')
    # pca = PCA(n_components=50)
    # vlatents = pca.fit_transform(vlatents)
    # print(vlatents.shape)
    # timer.stop(message='PCA done')
    # print('TSNE ...')
    # X_embedded = TSNE(n_components=2,
    #                   init='random',
    #                   perplexity=3,
    #                   learning_rate='auto',
    #                   early_exaggeration=24.,
    #                   n_iter=5000).fit_transform(vlatents)
    # print('TSNE done')
    print('UMAP ...')
    X_embedded = UMAP(
        n_components=2,
        init='random',
    ).fit_transform(vlatents)
    timer.stop(message='UMAP done')
    # print('MDS ...')
    # X_embedded = MDS(n_components=2).fit_transform(vlatents)
    # print('MDS done')
    print('Plotting ...')
    for label in np.unique(scopeclasses):
        sel = scopeclasses == label
        X_label = X_embedded[sel]
        plt.scatter(X_label[:, 0], X_label[:, 1], label=label, alpha=.5)
    plt.legend()
    timer.stop(message='Plotting done')
    plt.show()
    return X_embedded


if __name__ == '__main__':
    import sys
    import doctest
    import argparse
    # ### UNCOMMENT FOR LOGGING ####
    # import os
    # import logging
    # logfilename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    # logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    # logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-a', '--arg1')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()

    vlatents, scopeclasses = embed_scope()
    print(vlatents.shape)
    np.savez('latent_scope.npz', vlatents=vlatents, scopeclasses=scopeclasses)
    plot_latent(vlatents, scopeclasses, subsample=None, level=2)
