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

import misc.protein.sscl.encoder as encoder
import misc.protein.sscl.utils as utils
import misc.protein.get_pdb_title as get_pdb_title
import torch
import PDBloader
import faiss
import os
from misc.eta import ETA
import time
import numpy as np
import datetime
import matplotlib.pyplot as plt
from misc import kabsch
import pymol
from pymol import cmd
from misc.sequences.sequence_identity import seqalign
from collections import namedtuple
from prettytable import PrettyTable

cmd.set('fetch_type_default', 'mmtf')
cmd.set('fetch_path', cmd.exp_path('~/pdb'))


class Align():
    def __init__(self, pdb1, pdb2, model, sel1='all', sel2='all', latent_dims=512, feature_threshold=0.5, gap=-1.):
        """
        >>> pdb1 = '1ycr'
        >>> pdb2 = '7ad0'
        >>> sel1 = 'chain A'
        >>> sel2 = 'chain E'
        >>> model = encoder.load_model('models/sscl_fcn_20220615_2221.pt')
        Loading FCN model
        >>> aln = Align(pdb1=pdb1, pdb2=pdb2, model=model, sel1=sel1, sel2=sel2)
        >>> aln.similarity
        0.9920978546142578
        >>> aln.important_features()
        [(477, 0.20763037), (382, 0.19166075)]
        >>> aln.substitution_matrix().shape
        (85, 87)
        >>> aln.score_mat().shape
        (85, 87)

        # >>> _ = plt.matshow(aln.score_mat)
        # >>> _ = plt.colorbar()
        # >>> _ = plt.show()

        >>> aln.aln1
        {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 26: 25, 27: 26, 28: 27, 29: 28, 30: 29, 31: 30, 32: 31, 33: 32, 34: 33, 35: 34, 36: 35, 37: 36, 38: 37, 39: 38, 40: 39, 41: 40, 42: 41, 43: 42, 44: 43, 45: 44, 46: 45, 47: 46, 48: 47, 49: 48, 50: 49, 51: 50, 52: 51, 53: 52, 54: 53, 55: 54, 56: 55, 57: 56, 58: 57, 59: 58, 60: 59, 61: 60, 62: 61, 63: 62, 64: 63, 65: 64, 66: 65, 67: 66, 68: 67, 69: 68, 70: 69, 71: 70, 72: 71, 73: 72, 74: 73, 75: 74, 76: 75, 77: 76, 78: 77, 79: 78, 80: 79, 81: 80, 82: 81}
        >>> aln.aln2
        {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 23, 23: 24, 24: 25, 25: 26, 26: 27, 27: 28, 28: 29, 29: 30, 30: 31, 31: 32, 32: 33, 33: 34, 34: 35, 35: 36, 36: 37, 37: 38, 38: 39, 39: 40, 40: 41, 41: 42, 42: 43, 43: 44, 44: 45, 45: 46, 46: 47, 47: 48, 48: 49, 49: 50, 50: 51, 51: 52, 52: 53, 53: 54, 54: 55, 55: 56, 56: 57, 57: 58, 58: 59, 59: 60, 60: 61, 61: 62, 62: 63, 63: 64, 64: 65, 65: 66, 66: 67, 67: 68, 68: 69, 69: 70, 70: 71, 71: 72, 72: 73, 73: 74, 74: 75, 75: 76, 76: 77, 77: 78, 78: 79, 79: 80, 80: 81, 81: 82}

        >>> aln.structalign()
        """
        self.pdb1 = pdb1
        self.pdb2 = pdb2
        self.sel1 = sel1
        self.sel2 = sel2
        self.coords1 = utils.get_coords(pdb1, sel=sel1)
        self.coords2 = utils.get_coords(pdb2, sel=sel2)
        self.feature_threshold = feature_threshold
        self.gap = gap
        self.z1, self.conv1, self.dmat1 = encode(coords=self.coords1,
                                                 model=model,
                                                 latent_dims=latent_dims,
                                                 sel=sel1,
                                                 return_dmat=True)
        self.z2, self.conv2, self.dmat2 = encode(coords=self.coords2,
                                                 model=model,
                                                 latent_dims=latent_dims,
                                                 sel=sel2,
                                                 return_dmat=True)
        self.conv1 = self.conv1.cpu().detach().numpy().squeeze()
        self.conv2 = self.conv2.cpu().detach().numpy().squeeze()
        self.dmat1 = self.dmat1.detach().cpu().numpy().squeeze()
        self.dmat2 = self.dmat2.detach().cpu().numpy().squeeze()
        self.similarity = float(torch.matmul(self.z1, self.z2.T).squeeze().numpy())
        self.feature_importance = (self.z1 * self.z2).squeeze().numpy() / self.similarity
        self.zsub = self.substitution_matrix()
        self.M = self.score_mat()
        self.aln1, self.aln2 = self.traceback()

    def important_features(self):
        return get_important_features(self.feature_importance, threshold=self.feature_threshold)

    def substitution_matrix(self):
        """
        Get residue anchors for each latent dim
        """
        latent_dim, n1, _ = self.conv1.shape
        _, n2, _ = self.conv2.shape
        out = []
        zub = np.zeros((n1, n2))
        for ind in range(latent_dim):
            _, i1, j1 = np.unravel_index(self.conv1[ind, ...].argmax(), self.conv1.shape)
            _, i2, j2 = np.unravel_index(self.conv2[ind, ...].argmax(), self.conv2.shape)
            fi = self.feature_importance[ind]
            out.append(((i1, j1), (i2, j2), fi))
            zub[i1, i2] = max(fi, zub[i1, i2])
            zub[j1, j2] = max(fi, zub[j1, j2])
        return zub

    def score_mat(self):
        """
        """
        n1, n2 = self.zsub.shape
        M = np.zeros((n1, n2))
        for i in range(1, n1):
            for j in range(1, n2):
                M[i, j] = max(0, M[i - 1, j - 1] + self.zsub[i, j], M[i - 1, j] + self.gap, M[i, j - 1] + self.gap)
        return M

    def traceback(self):
        gap = self.gap
        n1, n2 = self.M.shape
        aln1 = dict()
        aln2 = dict()
        i, j = np.unravel_index(self.M.argmax(), self.M.shape)
        aln1[i] = j
        aln2[j] = i
        while i > 0 and j > 0:
            if self.M[i, j] == self.M[i - 1, j - 1] + self.zsub[i, j]:
                i = i - 1
                j = j - 1
                aln1[i] = j
                aln2[j] = i
            elif self.M[i, j] == self.M[i - 1, j] + gap:
                i = i - 1
                aln1[i] = None
            elif self.M[i, j] == self.M[i, j - 1] + gap:
                j = j - 1
                aln2[j] = None
            else:
                break
        aln1 = {k: aln1[k] for k in reversed(aln1)}
        aln2 = {k: aln2[k] for k in reversed(aln2)}
        return aln1, aln2

    def structalign(self, save_pse=True):
        s1 = [k for k in self.aln1 if self.aln1[k] is not None]
        s2 = [self.aln1[k] for k in self.aln1 if self.aln1[k] is not None]
        c1 = self.coords1[s1].numpy()
        c2 = self.coords2[s2].numpy()
        R, t = kabsch.rigid_body_fit(c1, c2)
        c1_aligned = (R.dot(c1.T)).T + t
        Metrics = namedtuple('Metrics', 'rmsd gdt')
        rmsd = utils.get_rmsd(c1_aligned, c2)
        gdt = utils.get_gdt(c1_aligned, c2)
        metrics = Metrics(rmsd, gdt)
        if save_pse:
            cmd.remove('all')
            try:
                cmd.load(filename=self.pdb1, object='p1')
            except pymol.CmdException:
                cmd.fetch(code=self.pdb1, name='p1')
            try:
                cmd.load(filename=self.pdb2, object='p1')
            except pymol.CmdException:
                cmd.fetch(code=self.pdb2, name='p2')
            cmd.remove(selection=f'not ({self.sel1}) and p1')
            cmd.remove(selection=f'not ({self.sel2}) and p2')
            toalign = cmd.get_coords('p1')
            coords_aligned = (R.dot(toalign.T)).T + t
            cmd.load_coords(coords_aligned, 'p1')
            cmd.orient()
            cmd.save('aln.pse')
        return metrics

    def plot(self):
        plot_sim(self.dmat1, self.dmat2, self.conv1, self.conv2, self.z1, self.z2, self.similarity)


def encode(model, pdb=None, coords=None, sel='all', latent_dims=512, return_dmat=False):
    """
    >>> model = encoder.load_model('models/sscl_fcn_20220610_1353.pt')
    Loading FCN model
    >>> z, conv = encode(pdb='pdb/yc/pdb1ycr.ent.gz', model=model)
    >>> z.shape
    torch.Size([1, 512])
    >>> conv.shape
    torch.Size([1, 512, 98, 98])
    """
    if pdb is not None:
        coords = utils.get_coords(pdb, sel=sel)
    dmat = utils.get_dmat(coords[None, ...])
    with torch.no_grad():
        try:
            # FCN model
            z, conv = model(dmat, get_conv=True)
        except TypeError:
            # CNN model
            z = model(dmat)
            conv = None
    if return_dmat:
        return z, conv, dmat
    else:
        return z, conv


def plot_sim(dmat1, dmat2, conv1, conv2, z1, z2, sim, threshold=8.):
    fig = plt.figure(constrained_layout=True)

    def onclick(event):
        for plotid, ax in enumerate(axes):
            if ax == event.inaxes:
                break
        # print(conv1.shape)  # (512, 226, 226)
        if plotid == 4:
            ind = int(event.xdata)
            maxconv1 = conv1[ind, ...].max()
            maxconv2 = conv2[ind, ...].max()
            _, i1, j1 = np.unravel_index(conv1[ind, ...].argmax(), conv1.shape)
            _, i2, j2 = np.unravel_index(conv2[ind, ...].argmax(), conv2.shape)
            print(ind, z1[ind], z2[ind], maxconv1, maxconv2, plotid)
            ax2.matshow(conv1[ind, ...])
            ax3.matshow(conv2[ind, ...])
            ax0.scatter(j1, i1)
            ax1.scatter(j2, i2)
            ax4.scatter(ind, z1[ind], zorder=2)
            fig.canvas.draw_idle()
        if plotid == 0 or plotid == 1:
            if plotid == 0:
                i1, j1 = int(event.ydata), int(event.xdata)
                ax0.scatter(j1, i1)
                ind1 = conv1[:, i1, j1].argmax()
                ind = ind1
                _, i2, j2 = np.unravel_index(conv2[ind1, ...].argmax(), conv2.shape)
                ax1.scatter(j2, i2)
            elif plotid == 1:
                i2, j2 = int(event.ydata), int(event.xdata)
                ax1.scatter(j2, i2)
                ind2 = conv2[:, i2, j2].argmax()
                ind = ind2
                _, i1, j1 = np.unravel_index(conv1[ind2, ...].argmax(), conv1.shape)
                ax0.scatter(j1, i1)
            # print(cmap1.shape)  # torch.Size([226, 226])
            ax2.matshow(conv1[ind, ...])
            ax3.matshow(conv2[ind, ...])
            ax4.scatter(ind, z1[ind], zorder=2)
            fig.canvas.draw_idle()

    z1 = z1.detach().cpu().numpy().squeeze()
    z2 = z2.detach().cpu().numpy().squeeze()
    # conv1 = conv1.cpu().detach().numpy().squeeze()
    # conv2 = conv2.cpu().detach().numpy().squeeze()
    cmap1 = utils.get_cmap(torch.tensor(dmat1))
    cmap2 = utils.get_cmap(torch.tensor(dmat2))
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    spec = fig.add_gridspec(3, 2)
    axes = []
    ax0 = fig.add_subplot(spec[0, 0])
    axes.append(ax0)
    ax0.matshow(cmap1)
    ax1 = fig.add_subplot(spec[0, 1])
    axes.append(ax1)
    ax1.matshow(cmap2)
    ax2 = fig.add_subplot(spec[1, 0])
    axes.append(ax2)
    ax2.matshow(conv1[0, ...])
    ax3 = fig.add_subplot(spec[1, 1])
    axes.append(ax3)
    ax3.matshow(conv2[0, ...])
    ax4 = fig.add_subplot(spec[2, :])
    axes.append(ax4)
    ax4.plot(z1)
    ax4.plot(z2)
    ax4.set_xlabel(f'latent_space: sim={sim:.4f}')
    plt.show()


def maxpool(conv):
    """
    >>> model = encoder.load_model('models/sscl_fcn_20220610_1353.pt')
    Loading FCN model
    >>> z, conv = encode(pdb='pdb/yc/pdb1ycr.ent.gz', model=model)
    >>> conv.shape
    torch.Size([1, 512, 98, 98])
    >>> out = maxpool(conv)
    >>> out.shape
    (1, 512, 98, 98)
    """
    conv = conv.detach().cpu().numpy()
    n = conv.shape[-1]
    latent_dim = conv.shape[1]
    bs = conv.shape[0]
    conv = conv.reshape((bs, latent_dim, -1))
    inds = conv.argmax(axis=-1)
    # print(inds.shape)  # (1, 512)
    out = np.zeros((bs, latent_dim, n * n))
    for b in range(bs):
        for la in range(latent_dim):
            i = inds[b, la]
            out[b, la, i] = conv[b, la, i]
    out = out
    out = out.reshape((bs, latent_dim, n, n))
    return out


def get_important_features(feature_importance, threshold=0.5):
    """
    """
    important_features = feature_importance.argsort()[::-1]
    csum = np.cumsum(feature_importance[important_features])
    important_features = important_features[csum <= threshold]
    importances = feature_importance[important_features]
    return list(zip(important_features, importances))


def build_index(pdblistfile,
                model,
                latent_dims=512,
                batch_size=4,
                do_break=np.inf,
                save_each=10,
                indexfilename='index.faiss'):
    """
    >>> model = encoder.load_model('models/sscl_fcn_20220610_1353.pt')
    Loading FCN model
    >>> build_index('pdbfilelist.txt', model, do_break=3, indexfilename='index_test.faiss')
    Total number of pdb in the FAISS index: 12
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        os.mkdir(indexfilename)
    except FileExistsError:
        pass
    dataset = PDBloader.PDBdataset(pdblistfile=pdblistfile, return_name=True, do_fragment=False)
    model = model.to(device)
    model.eval()
    index = faiss.IndexFlatIP(latent_dims)  # build the index in cosine similarity (IP: Inner Product)
    num_workers = os.cpu_count()
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             collate_fn=PDBloader.collate_fn)
    names = []
    eta = ETA(total_steps=len(dataloader))
    t_0 = time.time()
    for i, data in enumerate(dataloader):
        if i >= do_break:
            break
        batch = [dmat.to(device) for dmat, name in data if dmat is not None]
        try:
            with torch.no_grad():
                latent_vectors = torch.cat([model(e) for e in batch])
            # print(latent_vectors.shape)  # torch.Size([4, 512])
            latent_vectors = latent_vectors.detach().cpu().numpy()
            index.add(latent_vectors)
            names.extend([name for dmat, name in data if dmat is not None])
        except RuntimeError:
            print('System too large to fit in memory:')
            print([name for dmat, name in data if dmat is not None])
            print([dmat.shape for dmat, name in data if dmat is not None])
            pass
        eta_val = eta(i + 1)
        if (time.time() - t_0) / 60 >= save_each:
            t_0 = time.time()
            faiss.write_index(index, f'{indexfilename}/index.faiss')
            np.save(f'{indexfilename}/ids.npy', np.asarray(names))
        last_saved = (time.time() - t_0)
        last_saved = str(datetime.timedelta(seconds=last_saved))
        log(f"step: {i+1}|last_saved: {last_saved}|eta: {eta_val}")
    npdb = index.ntotal
    print(f'Total number of pdb in the FAISS index: {npdb}')
    faiss.write_index(index, f'{indexfilename}/index.faiss')
    names = np.asarray(names)
    np.save(f'{indexfilename}/ids.npy', names)


def print_foldsearch_results(Imat,
                             Dmat,
                             query_names,
                             ids,
                             model=None,
                             return_name=False,
                             return_seq_identity=False,
                             return_pdb_link=False,
                             return_rmsd=False,
                             return_gdt=False):
    table = PrettyTable()
    field_names = ['code', 'chain', 'similarity']
    if return_pdb_link:
        field_names.append('link')
    if return_seq_identity:
        field_names.append('seq_id')
    if return_name:
        field_names.append('title')
    if return_rmsd:
        field_names.append('RMSD (Å)')
    table.field_names = field_names
    table.float_format = '.4'
    for ind, dist, query in zip(Imat, Dmat, query_names):
        result_pdb_list = ids[ind]
        print(f'query: {query}')
        for pdb, d in zip(result_pdb_list, dist):
            # print(pdb)  # pdb/hf/pdb4hfz.ent.gz_A
            pdbcode = os.path.basename(pdb)
            pdbcode = os.path.splitext(os.path.splitext(pdbcode)[0])[0][-4:]
            chain = pdb.split('_')[1]
            row = [pdbcode, chain, float(d)]
            if return_pdb_link:
                row.append(f' https://www.rcsb.org/structure/{pdbcode}')
            if return_seq_identity:
                alignment, sequence_identity = seqalign.align(query, f'{pdbcode}_{chain}')
                row.append(f' {sequence_identity:.4f}')
            if return_name:
                title = get_pdb_title.get_pdb_title(pdbcode, chain)
                row.append(f' {title}')
            if return_rmsd or return_gdt:
                align = Align(pdb1=query, pdb2=f'{pdbcode}_{chain}', model=model)
                metric = align.structalign(save_pse=False)
            if return_rmsd:
                row.append(float(metric.rmsd))
            table.add_row(row)
    print(table)


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir


if __name__ == '__main__':
    import sys
    import doctest
    import argparse
    # ### UNCOMMENT FOR LOGGING ####
    # import os
    import logging
    logfilename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--similarity',
                        help='Compute the latent similarity between the 2 given pdb',
                        nargs=2,
                        metavar='file.pdb')
    parser.add_argument('--plotsim', help='Display interactive latent similarity plot', action='store_true')
    parser.add_argument(
        '-q',
        '--query',
        help='Search for nearest neighbors for the given query pdb. The query pdb should be given as {PDBCODE}_{CHAINID}'
    )
    parser.add_argument('--title', help='Retrieve PDB title information', action='store_true')
    parser.add_argument('--link', help='Display pdb link', action='store_true')
    parser.add_argument('--pid', help='Display sequence identity between match and query', action='store_true')
    parser.add_argument('--rmsd', help='Display RMSD between match and query', action='store_true')
    parser.add_argument('-n', help='Number of neighbors to return', type=int, default=5)
    parser.add_argument('--sel',
                        help='Selection for pdb file. Give two selections for the similarity computation',
                        default=None,
                        nargs='+')
    parser.add_argument('--model',
                        help='SSCL model to use',
                        metavar='model.pt',
                        default=f'{GetScriptDir()}/models/sscl_fcn_20220615_2221.pt')
    parser.add_argument('--latent_dims', default=512, type=int)
    parser.add_argument('--build_index', help='Build the FAISS index', action='store_true')
    parser.add_argument('--save_every',
                        help='Save the FAISS index every given number of minutes when building it',
                        type=int,
                        default=10)
    parser.add_argument(
        '--pdblist',
        help='File containing the list of pdb file to put in the index. Line format of the file: pdbfile chain')
    parser.add_argument('--bs', help='Batch size', type=int, default=4)
    parser.add_argument('--index',
                        help='FAISS index directory. Default: index_fcn_20220615_2221.faiss',
                        default=f'{GetScriptDir()}/index_fcn_20220615_2221.faiss')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()

    if args.similarity is not None:
        pdb1 = args.similarity[0]
        pdb2 = args.similarity[1]
        if args.sel is None:
            sel1 = 'all'
            sel2 = 'all'
        else:
            sel1 = args.sel[0]
            sel2 = args.sel[1]

        model = encoder.load_model(args.model, latent_dims=args.latent_dims)
        aln = Align(pdb1=pdb1, pdb2=pdb2, sel1=sel1, sel2=sel2, model=model, latent_dims=args.latent_dims)
        metrics = aln.structalign()
        if args.plotsim:
            aln.plot()
        print(f'similarity: {aln.similarity:.4f}')
        print(f'rmsd: {metrics.rmsd:.4f} Å')
        print(f'gdt: {metrics.gdt}')
    if args.query is not None:
        if args.sel is None:
            sel = 'all'
        else:
            sel = args.sel[0]
        model = encoder.load_model(args.model, latent_dims=args.latent_dims)
        z, conv = encode(pdb=args.query, sel=sel, model=model, latent_dims=args.latent_dims)
        z = z.detach().cpu().numpy()
        index = faiss.read_index(f'{args.index}/index.faiss')
        ids = np.load(f'{args.index}/ids.npy')
        Dmat, Imat = index.search(z, args.n)
        print_foldsearch_results(Imat,
                                 Dmat, [f'{args.query}'],
                                 ids,
                                 model=model,
                                 return_name=args.title,
                                 return_seq_identity=args.pid,
                                 return_pdb_link=args.link,
                                 return_rmsd=args.rmsd)
    if args.build_index:
        model = encoder.load_model(args.model, latent_dims=args.latent_dims)
        build_index(args.pdblist,
                    model,
                    latent_dims=args.latent_dims,
                    batch_size=args.bs,
                    save_each=args.save_every,
                    indexfilename=args.index)
