#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2023 Institut Pasteur                                       #
#############################################################################

import logging
import os

import numpy as np
import pymol2
import scipy.spatial.distance as scidist
import torch
from misc.mols.graph import mol_to_graph
from torch_geometric.data import Batch, Data

LOG = False

if LOG:
    if not os.path.isdir("logs"):
        os.mkdir("logs")
    logfilename = "logs/" + \
        os.path.splitext(os.path.basename(__file__))[0] + ".log"
    logging.basicConfig(filename=logfilename,
                        level=logging.INFO,
                        format="%(asctime)s: %(message)s")
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
            if LOG:
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
    onehot = torch.nn.functional.one_hot(classlist,
                                         num_classes=len(AMINO_ACIDS))
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


def mask_atom(node_features):
    n_atoms = node_features.shape[0]
    masked_atom_id = np.random.choice(n_atoms)
    masked_features = torch.clone(node_features[masked_atom_id])
    node_features[masked_atom_id] = torch.zeros_like(masked_features)
    return node_features, masked_features, masked_atom_id


def prot_to_graph(pdb,
                  extrafile=None,
                  selection=None,
                  masked_atom=False,
                  d_threshold=5.0):
    """
    - pdb: main pdb file to load. The pymol object name is myprot
    - extrafile: extra pdb file to load. The pymol object name is extra

    >>> node_features, edge_index, edge_features = prot_to_graph('data/1t4e.pdb')
    >>> node_features.shape
    torch.Size([784, 58])
    >>> edge_index.shape
    torch.Size([2, 18610])
    >>> edge_features.shape
    torch.Size([18610, 1])

    The extrafile can be used to select a pocket around a ligand
    Below the pocket is defined as all residues around 6 A of the ligand
    >>> node_features, edge_index, edge_features = prot_to_graph('data/1t4e.pdb', extrafile='data/lig.pdb', selection='byres(polymer.protein and (extra around 6))')
    >>> node_features.shape
    torch.Size([231, 58])
    >>> edge_index.shape
    torch.Size([2, 3907])
    >>> edge_features.shape
    torch.Size([3907, 1])

    Masking atoms
    >>> node_features, edge_index, edge_features, masked_features, masked_atom_id = prot_to_graph('data/1t4e.pdb', extrafile='data/lig.pdb', selection='byres(polymer.protein and (extra around 6))', masked_atom=True)
    >>> masked_features.shape
    torch.Size([58])
    >>> (masked_features == 0).all()
    tensor(False)
    >>> node_features[masked_atom_id].shape
    torch.Size([58])
    >>> (node_features[masked_atom_id]==0).all()
    tensor(True)
    >>> (node_features[masked_atom_id-1]==0).all()
    tensor(False)
    """
    if LOG:
        log(f"pdbfile: {pdb}")
    if selection is None:
        selection = "polymer.protein"
    with pymol2.PyMOL() as p:
        p.cmd.load(pdb, "myprot")
        if extrafile is not None:
            p.cmd.load(extrafile, "extra")
        p.cmd.remove("hydrogens")
        p.cmd.remove("resn hoh")
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
    resnames_onehot = seq_to_1hot(resnames)
    atomnames_onehot = atomlist_to_1hot(atomnames)
    node_features = torch.cat((resnames_onehot, atomnames_onehot), dim=1)
    #      58   =                 21          +      37
    if masked_atom:
        node_features, masked_features, masked_atom_id = mask_atom(
            node_features)
    dmat = scidist.squareform(scidist.pdist(coords))
    edge_index = torch.tensor(
        np.asarray(np.where(dmat < d_threshold)
                   ))  # edge_index has shape [2, E] with E the number of edges
    edge_features = 1.0 - \
        torch.tensor(dmat[tuple(edge_index)])[:, None] / d_threshold
    if not masked_atom:
        return (
            node_features.to(torch.float32),
            edge_index,
            edge_features.to(torch.float32),
        )
    else:
        return (
            node_features.to(torch.float32),
            edge_index,
            edge_features.to(torch.float32),
            masked_features.float(),
            masked_atom_id,
        )


class Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        txtfile,
        radius=6.0,
        return_pyg_graph=False,
        masked_atom=False,
        return_mol_graph=False,
    ):
        """
        - txtfile: a txtfile containing the list of data. The format is the following:

            key path/to/pdb/protein/file.pdb optional/path/to/ligand/file.sdf

            - key is any string
            - path/to/pdb/protein/file.pdb is the protein pdb file to pass to prot_to_graph
            - optional/path/to/ligand/file.sdf is optional. It give the path to a ligand file to optionaly select a
              pocket in path/to/pdb/protein/file.pdb

        - radius: radius in Angstrom to optionnaly define a protein pocket around a ligand given in txtfile

        >>> dataset = Dataset('data/dude_test_100.smi')
        >>> dataset.__len__()
        97
        >>> key, (node_features, edge_index, edge_features) = dataset[3]
        >>> key
        ('COc4ccc(S(=O)(=O)N(Cc1cccnc1)c2c(C(=O)NO)cnc3c(Br)cccc23)cc4', 'data/DUDE100/mmp13/receptor.pdb', 'data/DUDE100/mmp13/crystal_ligand.sdf')
        >>> node_features.shape
        torch.Size([192, 58])
        >>> edge_index.shape
        torch.Size([2, 3438])
        >>> edge_features.shape
        torch.Size([3438, 1])

        >>> dataset = Dataset('data/dude_test_100.smi', return_pyg_graph=True)
        >>> graph = dataset[3]
        >>> graph
        Data(x=[192, 58], edge_index=[2, 3438], edge_attr=[3438, 1], y=[3])
        >>> print(graph.batch)
        None

        Try a dataloader:
        >>> from torch_geometric.loader import DataLoader
        >>> loader = DataLoader(dataset, batch_size=8)
        >>> for batch in loader:
        ...     break
        >>> batch
        DataBatch(x=[1710, 58], edge_index=[2, 28294], edge_attr=[28294, 1], y=[8], batch=[1710], ptr=[9])
        >>> batch.y
        [('Cn3c(=O)c(c1c(Cl)cccc1Cl)cc4cnc(NCCCN2CCOCC2)cc34', 'data/DUDE100/src/receptor.pdb', 'data/DUDE100/src/crystal_ligand.sdf'), ...

        Get the index to split back the batch
        >>> batch.batch
        tensor([0, 0, 0,  ..., 7, 7, 7])

        Try with masked atoms
        >>> dataset = Dataset('data/dude_test_100.smi', return_pyg_graph=True, masked_atom=True)
        >>> dataset[0]
        Data(x=[227, 58], edge_index=[2, 3523], edge_attr=[3523, 1], y=[3], masked_features=[58], masked_atom_id=...)
        >>> loader = DataLoader(dataset, batch_size=8)
        >>> for batch in loader:
        ...     break
        >>> batch
        DataBatch(x=[1710, 58], edge_index=[2, 28294], edge_attr=[28294, 1], y=[8], masked_features=[464], masked_atom_id=[8], batch=[1710], ptr=[9])

        Try to return the molgraph
        >>> dataset = Dataset('data/dude_test_100.smi', return_pyg_graph=True, return_mol_graph=True)
        >>> graph = dataset[3]
        >>> graph
        Data(x=[192, 58], edge_index=[2, 3438], edge_attr=[3438, 1], y=[3], molgraph=Data(x=[53, 16], edge_index=[2, 112], edge_attr=[112, 4], pos=[53, 3], edge_type=[112]))
        >>> loader = DataLoader(dataset, batch_size=8)
        >>> for batch in loader:
        ...     break
        >>> batch
        DataBatch(x=[1710, 58], edge_index=[2, 28294], edge_attr=[28294, 1], y=[8], molgraph=[8], batch=[1710], ptr=[9])

        Create a batch of molgraph from batch.molgraph
        >>> Batch.from_data_list(batch.molgraph)
        DataBatch(x=[438, 16], edge_index=[2, 922], edge_attr=[922, 4], pos=[438, 3], edge_type=[922], batch=[438], ptr=[9])

        """
        self.txtfile = txtfile
        self.radius = radius
        self.len = 0
        self.return_pyg_graph = return_pyg_graph
        self.shelve = dict()
        self.read_txtfile()
        self.masked_atom = masked_atom
        self.return_mol_graph = return_mol_graph

    def read_txtfile(self):
        with open(self.txtfile, "r") as inp:
            i = 0
            for line in inp:
                if line.startswith("#"):
                    continue
                data = line.split()
                self.shelve[i] = data
                self.len += 1
                i += 1

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        data = self.shelve[index]
        label = data[0]
        value = data[1:]
        protfilename = value[0]
        ligand = value[1]
        key = (label, protfilename, ligand)
        if self.return_mol_graph:
            # label must be a smiles
            molgraph = mol_to_graph.smiles_to_graph(label)
        else:
            molgraph = None
        if len(value) > 1:
            ligfilename = value[1]
            selection = f"byres(polymer.protein and (extra around {self.radius}))"
        else:
            ligfilename = None
            selection = None
        graph_features = prot_to_graph(
            protfilename,
            extrafile=ligfilename,
            selection=selection,
            masked_atom=self.masked_atom,
        )
        node_features, edge_index, edge_features = graph_features[:3]
        if self.masked_atom:
            masked_features, masked_atom_id = graph_features[-2:]
        else:
            masked_features = None
            masked_atom_id = None
        if not self.return_pyg_graph:
            return key, (node_features, edge_index, edge_features)
        else:
            return Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_features,
                y=key,
                masked_features=masked_features,
                masked_atom_id=masked_atom_id,
                molgraph=molgraph,
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
    import argparse
    import doctest
    import sys

    # ### UNCOMMENT FOR LOGGING ####
    # import os
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument("-p", "--pdb")
    parser.add_argument("--test", help="Test the code", action="store_true")
    parser.add_argument("--func",
                        help="Test only the given function(s)",
                        nargs="+")
    args = parser.parse_args()

    # If log is present log the arguments to the log file:
    if LOG:
        for k, v in args._get_kwargs():
            log(f"# {k}: {v}")

    if args.test:
        if args.func is None:
            doctest.testmod(optionflags=doctest.ELLIPSIS
                            | doctest.REPORT_ONLY_FIRST_FAILURE)
        else:
            for f in args.func:
                print(f"Testing {f}")
                f = getattr(sys.modules[__name__], f)
                doctest.run_docstring_examples(
                    f,
                    globals(),
                    optionflags=doctest.ELLIPSIS
                    | doctest.REPORT_ONLY_FIRST_FAILURE,
                )
        sys.exit()
    if args.pdb is not None:
        prot_to_graph(args.pdb)
