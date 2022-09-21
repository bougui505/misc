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
import gzip
import faiss
import numpy as np
import tqdm
import time
import scipy.spatial.distance as scidist
from misc import list_utils


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir


def load_index(index_dir):
    '''
    >>> findex, ids = load_index('index.faiss')
    >>> ids
    {'200l_A': 0, '101m_A': 1, '102m_A': 2, '102l_A': 3, '105m_A': 4, '107m_A': 5, '104m_A': 6, ...
    >>> findex
    <faiss.swigfaiss.IndexFlat; proxy of <Swig Object of type 'faiss::IndexFlat *' at ...> >

    # One can get the vector for the given id (e.g. 10):
    >>> type(findex.reconstruct(10))
    <class 'numpy.ndarray'>
    >>> findex.reconstruct(10).shape
    (512,)
    '''
    findex = faiss.read_index(f'{index_dir}/index.faiss')
    ids = np.load(f'{index_dir}/ids.npy')
    ids = {f'{e[10:14]}_{e[-1]}': i for i, e in enumerate(ids)}
    return findex, ids


def read_scope40(scopefile):
    """
    >>> homologs_lists = read_scope40('data/homologs_scope40.txt.gz')
    >>> type(homologs_lists)
    <class 'list'>
    >>> len(homologs_lists)
    8820
    >>> homologs_lists[10]
    ['1a1i_A', '1a1k_A', '1a1j_A', '1jk2_A', ...
    """
    with gzip.open(scopefile, 'r') as scopefile:
        homologs_list = []
        for line in scopefile:
            line = line.decode().strip()
            homologs_list.append(line.split())
    return homologs_list


def get_all_ranks(homologs_lists, findex, ids, outfile=None):
    """
    >>> homologs_lists = read_scope40('data/homologs_scope40.txt.gz')
    >>> findex, ids = load_index('index.faiss')
    >>> ids
    {'200l_A': 0, '101m_A': 1, '102m_A': 2, '102l_A': 3, '105m_A': 4, '107m_A': 5, '104m_A': 6, ...
    >>> ranks, distances = get_all_ranks(homologs_lists[:10], findex, ids, outfile='test.txt')

    >>> ranks
    [1, 2, 3, 4709, 2145, 2187, 2377, 2456, 2664, 2784, 2905, 3108, 3160, 3174, ...

    >>> distances
    [0.9999725, 0.9999109, 0.9997288, 0.9328714, 0.9242841, 0.9238974, 0.92215693, ...
    """
    nested_homologs = [[str(e) for e in homologs if e in ids] for homologs in homologs_lists]
    nested_list = [[ids[str(e)] for e in homologs] for homologs in nested_homologs]
    z_anchors = []
    for li in nested_list:
        z_anchors.append(findex.reconstruct(li[0]))
    z_anchors = np.asarray(z_anchors, dtype=np.float32)
    dmat, indices = findex.search(z_anchors, 10000)
    ranks = []
    distances = []
    return_ids = []
    anchor_ids = []
    homologs_ids = []
    inverse_mapping = {v: k for (k, v) in ids.items()}
    for i, kneigh in enumerate(indices):
        ids_list = np.asarray(nested_list[i])
        ranks_i = np.where(np.isin(kneigh, ids_list))[0][1:]  # Remove the first element (z_a itself)
        return_ids.append(kneigh[ranks_i])
        ranks.append(list(ranks_i))
        distances.append(list(dmat[i][ranks_i]))
        anchor_ids.append([nested_homologs[i][0]] * len(ranks_i))
        homologs_ids.append([inverse_mapping[e] for e in return_ids[-1]])
    np.savez('data/homologs_scope40_ranks.npz', ids=return_ids, ranks=ranks, distances=distances)
    if outfile is not None:
        np.savetxt(outfile,
                   np.c_[list_utils.flatten(anchor_ids),
                         list_utils.flatten(homologs_ids),
                         list_utils.flatten(ranks),
                         list_utils.flatten(distances)],
                   fmt=('%s', '%s', '%s', '%s'),
                   header='#anchor #homolog #rank #sim',
                   comments='')
    return ranks, distances

    # all_ranks = []
    # all_distances = []
    # # pbar = tqdm.tqdm(total=len(homologs_lists))
    # for homologs in homologs_lists:
    #     ranks, distances = get_ranks(homologs, findex, ids)
    #     all_ranks.extend(list(ranks))
    #     all_distances.extend(list(distances))
    # #     pbar.update(1)
    # # pbar.close()
    # return all_ranks, all_distances


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
    homologs_lists = read_scope40('data/homologs_scope40.txt.gz')
    findex, ids = load_index('index.faiss')
    ranks, distances = get_all_ranks(homologs_lists, findex, ids, outfile='data/homologs_scope40_ranks.txt')
