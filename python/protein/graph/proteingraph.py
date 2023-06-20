#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2023 Institut Pasteur                                       #
#                               				                            #
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
import pymol2
import numpy as np
import torch
from torch_geometric.data import Data
import scipy.spatial.distance as scidist
from misc.shelve.tempshelve import Tempshelve
import logging

if not os.path.isdir("logs"):
    os.mkdir("logs")
logfilename = "logs/" + os.path.splitext(os.path.basename(__file__))[0] + ".log"
logging.basicConfig(
    filename=logfilename, level=logging.INFO, format="%(asctime)s: %(message)s"
)
logging.info(f"################ Starting {__file__} ################")

AMINO_ACIDS = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "XXX",
]

ATOMS = [
    "CE3",
    "CG1",
    "ND1",
    "N",
    "NE1",
    "CG2",
    "ND2",
    "CZ",
    "O",
    "CH2",
    "NE2",
    "C",
    "CA",
    "CB",
    "NH1",
    "NE",
    "OD1",
    "NH2",
    "CD",
    "CE",
    "OE1",
    "OD2",
    "CZ2",
    "OG",
    "OE2",
    "OH",
    "CZ3",
    "OG1",
    "SD",
    "CG",
    "CD1",
    "CE1",
    "CD2",
    "SG",
    "CE2",
    "NZ",
    "XXX",
]


def getclasslist(mapping, inplist):
    classlist = []
    for e in inplist:
        if e in mapping:
            classlist.append(mapping[e])
        else:
            classlist.append(mapping["XXX"])
            log(f"unknown key {e}")
    return torch.tensor(classlist)


def seq_to_1hot(seqlist):
    """
    >>> seqlist = ['GLY', 'SER', 'GLN', 'ILE', 'PRO', 'ALA', 'SER', 'GLU', 'GLN', 'GLU', 'DLY', 'THR', 'LEU']
    >>> len(seqlist)
    13
    >>> onehot = seq_to_1hot(seqlist)
    >>> onehot.shape
    torch.Size([13, 21])
    >>> onehot[0]
    tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> onehot[10]
    tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    """
    mapping = dict(zip(AMINO_ACIDS, range(len(AMINO_ACIDS))))
    classlist = getclasslist(mapping, seqlist)
    onehot = torch.nn.functional.one_hot(classlist, num_classes=len(AMINO_ACIDS))
    return onehot


def atomlist_to_1hot(atomlist):
    """
    >>> atomlist = ['ND1', 'CE3', 'CG1', 'Y', 'CA']
    >>> len(atomlist)
    5
    >>> onehot = atomlist_to_1hot(atomlist)
    >>> onehot.shape
    torch.Size([5, 37])
    >>> onehot[0]
    tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> onehot[3]
    tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    """
    mapping = dict(zip(ATOMS, range(len(ATOMS))))
    classlist = getclasslist(mapping, atomlist)
    onehot = torch.nn.functional.one_hot(classlist, num_classes=len(ATOMS))
    return onehot


def prot_to_graph(pdb, extrafile=None, selection=None, compute_sasa=False):
    """
    - pdb: main pdb file to load. The pymol object name is myprot
    - extrafile: extra pdb file to load. The pymol object name is extra

    >>> node_features, edge_index, edge_features, sasa = prot_to_graph('data/1t4e.pdb', compute_sasa=True)
    >>> node_features.shape
    torch.Size([784, 58])
    >>> edge_index.shape
    torch.Size([2, 10030])
    >>> edge_features.shape
    torch.Size([10030, 1])
    >>> sasa
    10119.4404296875

    The extrafile can be used to select a pocket around a ligand
    Below the pocket is defined as all residues around 6 A of the ligand
    >>> node_features, edge_index, edge_features, sasa = prot_to_graph('data/1t4e.pdb', extrafile='data/lig.pdb', selection='byres(polymer.protein and (extra around 6))', compute_sasa=True)
    >>> node_features.shape
    torch.Size([231, 58])
    >>> edge_index.shape
    torch.Size([2, 2363])
    >>> edge_features.shape
    torch.Size([2363, 1])
    >>> sasa
    2959.5849609375
    """
    log(f"pdbfile: {pdb}")
    if selection is None:
        selection = "polymer.protein"
    with pymol2.PyMOL() as p:
        p.cmd.load(pdb, "myprot")
        if extrafile is not None:
            p.cmd.load(extrafile, "extra")
        selection = f"({selection}) and myprot"
        coords = p.cmd.get_coords(selection=selection)
        space = {"resnames": [], "atomnames": []}
        p.cmd.iterate(
            selection=selection,
            space=space,
            expression="resnames.append(resn); atomnames.append(name)",
        )
        resnames = np.asarray(space["resnames"])
        atomnames = np.asarray(space["atomnames"])
        if compute_sasa:
            p.cmd.remove(selection="myprot and not polymer.protein")
            sasa = p.cmd.get_area(selection=selection)
        else:
            sasa = None
    resnames_onehot = seq_to_1hot(resnames)
    atomnames_onehot = atomlist_to_1hot(atomnames)
    node_features = torch.cat((resnames_onehot, atomnames_onehot), dim=1)
    dmat = scidist.squareform(scidist.pdist(coords))
    d_threshold = 4.0
    edge_index = torch.tensor(
        np.asarray(np.where(dmat < d_threshold))
    )  # edge_index has shape [2, E] with E the number of edges
    edge_features = 1.0 - torch.tensor(dmat[tuple(edge_index)])[:, None] / d_threshold
    return (
        node_features.to(torch.float32),
        edge_index,
        edge_features.to(torch.float32),
        sasa,
    )


class Dataset(torch.utils.data.Dataset):
    def __init__(self, txtfile, radius=6.0, return_pyg_graph=False, compute_sasa=False):
        """
        - txtfile: a txtfile containing the list of data. The format is the following:

            key path/to/pdb/protein/file.pdb optional/path/to/ligand/file.sdf

            - key is any string
            - path/to/pdb/protein/file.pdb is the protein pdb file to pass to prot_to_graph
            - optional/path/to/ligand/file.sdf is optional. It give the path to a ligand file to optionaly select a
              pocket in path/to/pdb/protein/file.pdb
            - compute_sasa: if True returns the SASA in the y feature of the graph when return_pyg_graph=True

        - radius: radius in Angstrom to optionnaly define a protein pocket around a ligand given in txtfile

        >>> dataset = Dataset('data/dude_test_100.smi')
        >>> dataset.__len__()
        97
        >>> key, (node_features, edge_index, edge_features), sasa = dataset[3]
        >>> key
        ('COc4ccc(S(=O)(=O)N(Cc1cccnc1)c2c(C(=O)NO)cnc3c(Br)cccc23)cc4', 'data/DUDE100/mmp13/receptor.pdb', 'data/DUDE100/mmp13/crystal_ligand.sdf')
        >>> node_features.shape
        torch.Size([242, 58])
        >>> edge_index.shape
        torch.Size([2, 3074])
        >>> edge_features.shape
        torch.Size([3074, 1])

        >>> dataset = Dataset('data/dude_test_100.smi', return_pyg_graph=True, compute_sasa=True)
        >>> graph = dataset[3]
        >>> graph
        Data(x=[242, 58], edge_index=[2, 3074], edge_attr=[3074, 1], y=[3], sasa=2694.87060546875)
        >>> print(graph.batch)
        None

        Try a dataloader:
        >>> from torch_geometric.loader import DataLoader
        >>> loader = DataLoader(dataset, batch_size=8)
        >>> for batch in loader:
        ...     break
        >>> batch
        DataBatch(x=[2534, 58], edge_index=[2, 31716], edge_attr=[31716, 1], y=[8], sasa=[8], batch=[2534], ptr=[9])
        >>> batch.y
        [('Cn3c(=O)c(c1c(Cl)cccc1Cl)cc4cnc(NCCCN2CCOCC2)cc34', 'data/DUDE100/src/receptor.pdb', 'data/DUDE100/src/crystal_ligand.sdf'), ...
        >>> batch.sasa
        tensor([3873.5662, 3873.5662, 3605.4141, 2694.8706, 3514.5415, 3902.6648,
                3083.0132, 2916.6514])

        Get the index to split back the batch
        >>> batch.batch
        tensor([0, 0, 0,  ..., 7, 7, 7])
        """
        self.txtfile = txtfile
        self.radius = radius
        self.len = 0
        self.return_pyg_graph = return_pyg_graph
        self.shelve = Tempshelve()
        self.read_txtfile()
        self.compute_sasa = compute_sasa

    def read_txtfile(self):
        with open(self.txtfile, "r") as inp:
            for i, line in enumerate(inp):
                data = line.split()
                self.shelve.add(str(i), data)
                self.len += 1

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        data = self.shelve.get(str(index))
        label = data[0]
        value = data[1:]
        protfilename = value[0]
        ligand = value[1]
        key = (label, protfilename, ligand)
        if len(value) > 1:
            ligfilename = value[1]
            selection = f"byres(polymer.protein and (extra around {self.radius}))"
        else:
            ligfilename = None
            selection = None
        node_features, edge_index, edge_features, sasa = prot_to_graph(
            protfilename,
            extrafile=ligfilename,
            selection=selection,
            compute_sasa=self.compute_sasa,
        )
        if not self.return_pyg_graph:
            return key, (node_features, edge_index, edge_features), sasa
        else:
            return Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_features,
                y=key,
                sasa=sasa,
            )


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir


if __name__ == "__main__":
    import sys
    import doctest
    import argparse

    # ### UNCOMMENT FOR LOGGING ####
    # import os
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument("-p", "--pdb")
    parser.add_argument("--test", help="Test the code", action="store_true")
    parser.add_argument("--func", help="Test only the given function(s)", nargs="+")
    args = parser.parse_args()

    # If log is present log the arguments to the log file:
    for k, v in args._get_kwargs():
        log(f"# {k}: {v}")

    if args.test:
        if args.func is None:
            doctest.testmod(
                optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE
            )
        else:
            for f in args.func:
                print(f"Testing {f}")
                f = getattr(sys.modules[__name__], f)
                doctest.run_docstring_examples(
                    f,
                    globals(),
                    optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE,
                )
        sys.exit()
    if args.pdb is not None:
        prot_to_graph(args.pdb)
