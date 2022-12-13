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
import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_scatter import scatter_max


class RGCN(torch.nn.Module):
    """
    >>> from misc.mols.graph.mol_to_graph import molfromsmiles, get_mol_graph, MolDataset
    >>> smiles = "O[C@@H]([C@H]1O[C@H]([C@H](O)[C@@H]1O)n1ccc2C3=NCC(O)N3C=Nc12)c1ccc(Cl)cc1"
    >>> mol = molfromsmiles(smiles)
    >>> mol.GetNumAtoms()
    48
    >>> graph = get_mol_graph(mol)
    >>> graph
    Data(x=[48, 16], edge_index=[2, 104], edge_attr=[104, 4], pos=[48, 3], edge_type=[104])
    >>> rgcn = RGCN(num_node_features=16, num_relations=4, maxpool=False)
    >>> parameters = rgcn.parameters()
    >>> num_trainable_params = sum(p.numel() for p in parameters)
    >>> num_trainable_params
    14192
    >>> rgcn.conv1.weight.shape
    torch.Size([4, 16, 16])
    >>> out = rgcn(graph)
    >>> out.shape
    torch.Size([48, 64])

    >>> rgcn = RGCN(num_node_features=16, num_relations=4, maxpool=True)
    >>> out = rgcn(graph)
    >>> out.shape
    torch.Size([64])

    Try on a batch:
    >>> from torch_geometric.loader import DataLoader
    >>> seed = torch.manual_seed(0)
    >>> smilesfilename = 'data/test.smi'
    >>> dataset = MolDataset(smilesfilename)
    >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
    >>> iterator = iter(loader)
    >>> batch = next(iterator)
    >>> batch
    DataBatch(x=[1867, 16], edge_index=[2, 3912], edge_attr=[3912, 4], pos=[1867, 3], edge_type=[3912], batch=[1867], ptr=[33])
    >>> rgcn = RGCN(num_node_features=16, num_relations=4, maxpool=False)
    >>> out = rgcn(batch)
    >>> out.shape
    torch.Size([1867, 64])
    >>> rgcn = RGCN(num_node_features=16, num_relations=4, maxpool=True)
    >>> out = rgcn(batch)
    >>> out.shape
    torch.Size([32, 64])
    """
    def __init__(self, num_node_features, num_relations, maxpool=False):
        super().__init__()
        self.conv1 = RGCNConv(in_channels=num_node_features, out_channels=16, num_relations=num_relations)
        self.conv2 = RGCNConv(in_channels=16, out_channels=32, num_relations=num_relations)
        self.conv3 = RGCNConv(in_channels=32, out_channels=64, num_relations=num_relations)
        self.maxpool = maxpool

    def forward(self, graph):
        x = graph.x
        x = self.conv1(x=x, edge_index=graph.edge_index, edge_type=graph.edge_type)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x=x, edge_index=graph.edge_index, edge_type=graph.edge_type)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x=x, edge_index=graph.edge_index, edge_type=graph.edge_type)
        if self.maxpool:
            if graph.batch is None:  # No batch
                x = torch.max(x, axis=0)[0]
            else:  # Take the max along the batch
                x = scatter_max(x, graph.batch, dim=0)[0]
        return x  # F.log_softmax(x, dim=1)


class MLP(torch.nn.Module):
    def __init__(self, num_input_features, hidden_units, num_classes):
        """
        >>> seed = torch.manual_seed(0)
        >>> embedding = torch.rand(32, 64)

        >>> mlp = MLP(64, [128], 12)
        >>> out = mlp(embedding)
        >>> out.shape
        torch.Size([32, 12])

        >>> mlp = MLP(64, [128, 256], 12)
        >>> out = mlp(embedding)
        >>> out.shape
        torch.Size([32, 12])
        """
        super().__init__()

        # Define the model layers
        self.fc1 = torch.nn.Linear(num_input_features, hidden_units[0])
        self.fc_hidden_layers = []
        if len(hidden_units) == 1:
            hidden_units += hidden_units
        for i, num_hidden_units in enumerate(hidden_units[:-1]):
            nout = hidden_units[i + 1]
            self.fc_hidden_layers.append(torch.nn.Linear(num_hidden_units, nout))
        self.fclast = torch.nn.Linear(nout, num_classes)

        # Define the activation functions
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # Apply the first linear layer and the ReLU activation function
        x = self.fc1(x)
        x = self.relu(x)

        # Apply the second linear layer and the ReLU activation function
        for fc_hidden in self.fc_hidden_layers:
            x = fc_hidden(x)
            x = self.relu(x)

        # Apply the third linear layer and the softmax activation function
        x = self.fclast(x)
        x = self.softmax(x)

        return x


class RGCNN(torch.nn.Module):
    """
    RGCN + MLP
    >>> from misc.mols.graph.mol_to_graph import molDataLoader
    >>> seed = torch.manual_seed(0)
    >>> loader = molDataLoader('data/HMT_mols_test.smi', readclass=True, reweight=True)
    >>> loader
    <torch_geometric.loader.dataloader.DataLoader object at ...
    >>> iterator = iter(loader)
    >>> batch = next(iterator)
    >>> batch
    [DataBatch(x=[1706, 16], edge_index=[2, 3590], edge_attr=[3590, 4], pos=[1706, 3], edge_type=[3590], batch=[1706], ptr=[33]), ('PRMT5', 'SMYD3', 'SETD2', 'PRDM6', 'MLL2', 'MLL', 'MLL', 'PRMT3', 'FBL', 'PCMT1', 'MLL', 'PRMT2', 'PRDM5', 'PRDM14', 'CARM1', 'NSD3', 'PRMT1', 'PRMT5', 'PRDM5', 'PCMT1', 'SMYD3', 'SUV39H1', 'SETDB1', 'MECOM', 'SETDB1', 'PRMT3', 'PRMT5', 'PRMT1', 'MECOM', 'MLL2', 'PRMT5', 'EZH1')]
    >>> x, y = batch
    >>> rgcnn = RGCNN(num_classes=26)
    >>> y_pred = rgcnn(x)
    >>> y_pred.shape
    torch.Size([32, 26])
    """
    def __init__(self, num_classes, num_node_features=16, num_relations=4, hidden_units=[128, 256]):
        super().__init__()
        self.rgcn = RGCN(num_node_features=num_node_features, num_relations=num_relations, maxpool=True)
        self.mlp = MLP(num_input_features=64, hidden_units=hidden_units, num_classes=num_classes)

    def forward(self, x):
        x = self.rgcn(x)
        x = self.mlp(x)
        return x


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
    # import logging
    # if not os.path.isdir('logs'):
    #     os.mkdir('logs')
    # logfilename = 'logs/' + os.path.splitext(os.path.basename(__file__))[0] + '.log'
    # logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    # logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-a', '--arg1')
    parser.add_argument('--test', help='Test the code', action='store_true')
    parser.add_argument('--func', help='Test only the given function(s)', nargs='+')
    args = parser.parse_args()

    # If log is present log the arguments to the log file:
    for k, v in args._get_kwargs():
        log(f'# {k}: {v}')

    if args.test:
        if args.func is None:
            doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        else:
            for f in args.func:
                print(f'Testing {f}')
                f = getattr(sys.modules[__name__], f)
                doctest.run_docstring_examples(f, globals())
        sys.exit()
