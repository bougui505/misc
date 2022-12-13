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
from misc.eta import ETA
from misc.mols.graph.mol_to_graph import molDataLoader
import time


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
    >>> seed = torch.manual_seed(0)
    >>> loader = molDataLoader('data/HMT_mols_test.smi', readclass=True, reweight=True)
    >>> loader
    <torch_geometric.loader.dataloader.DataLoader object at ...
    >>> iterator = iter(loader)
    >>> batch = next(iterator)
    >>> batch
    [DataBatch(x=[1951, 16], edge_index=[2, 4080], edge_attr=[4080, 4], pos=[1951, 3], edge_type=[4080], batch=[1951], ptr=[33]), tensor([34, 21,  8,  9,  1, 22, 19, 28,  5, 21, 22, 19,  1, 20, 21, 38, 13,  9,
             0, 13,  9, 10,  5, 19,  7, 28,  4, 37, 19, 18,  4,  5])]
    >>> x, y = batch
    >>> rgcnn = RGCNN(num_classes=51)
    >>> y_pred = rgcnn(x)
    >>> y_pred.shape
    torch.Size([32, 51])

    Test on GPU
    # >>> device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # >>> rgcnn = rgcnn.to(device)
    # >>> x = x.to(device)
    # >>> y_pred = rgcnn(x)
    # >>> y_pred.shape
    # torch.Size([32, 51])
    """
    def __init__(self, num_classes, num_node_features=16, num_relations=4, hidden_units=[128, 256]):
        super().__init__()
        self.rgcn = RGCN(num_node_features=num_node_features, num_relations=num_relations, maxpool=True)
        self.mlp = MLP(num_input_features=64, hidden_units=hidden_units, num_classes=num_classes)

    def forward(self, x):
        x = self.rgcn(x)
        x = self.mlp(x)
        return x


def metric(y_true, y_pred):
    """
    >>> seed = torch.manual_seed(0)
    >>> y_true = torch.randint(low=0, high=51, size=(8,))
    >>> y_true
    tensor([41,  3, 29, 33, 19,  0, 16, 10])
    >>> y_true[1] = 10  # Just for testing purpose
    >>> y_true[6] = 23  # Just for testing purpose
    >>> y_pred = torch.rand(size=(8, 51))
    >>> y_pred.shape
    torch.Size([8, 51])
    >>> metric(y_true, y_pred)
    0.25
    """
    _, class_pred = torch.max(y_pred, dim=1)
    return float((class_pred == y_true).sum() / len(y_true))


def test_model(model, testloader):
    """
    >>> seed = torch.manual_seed(0)
    >>> loader, testloader = molDataLoader('data/HMT_mols_test.smi', readclass=True, reweight=True, testset_len=8)
    >>> rgcnn = RGCNN(num_classes=51)
    >>> test_model(rgcnn, testloader)
    0.0
    """
    metric_val = 0
    with torch.no_grad():
        for batch in testloader:
            x, y_true = batch
            y_pred = model(x)
            metric_val += metric(y_true, y_pred)
    metric_val /= len(testloader)
    return metric_val


def train(smiles_filename, n_epochs, testset_len=128):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataloader, testloader = molDataLoader(smiles_filename, readclass=True, reweight=True, testset_len=testset_len)
    dataiter = iter(dataloader)
    model = RGCNN(num_classes=51)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters())
    loss = torch.nn.CrossEntropyLoss()
    epoch = 0
    step = 0
    total_steps = n_epochs * len(dataloader)
    t_0 = time.time()
    eta = ETA(total_steps=total_steps)
    for epoch in range(n_epochs):
        while True:
            try:
                batch = next(dataiter)
            except ValueError:
                print(f"Error for batch {step}")
                continue
            step += 1
            x, y_true = batch
            x = x.to(device)
            y_true = y_true.to(device)
            y_pred = model(x)
            loss_val = loss(y_pred, y_true)
            loss_val.backward()
            opt.step()
            opt.zero_grad()
            eta_val = eta(step)
            log(f"epoch: {epoch+1}|step: {step}/{total_steps}|loss: {loss_val}|eta: {eta_val}")


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
    ### UNCOMMENT FOR LOGGING ####
    import os
    import logging
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    logfilename = 'logs/' + os.path.splitext(os.path.basename(__file__))[0] + '.log'
    logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    logging.info(f"################ Starting {__file__} ################")
    ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--train', help='train the model', action='store_true')
    parser.add_argument('--smiles', help='SMILES filename')
    parser.add_argument('--nepochs', type=int, default=100)
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
    if args.train:
        train(smiles_filename=args.smiles, n_epochs=args.nepochs, testset_len=128)
