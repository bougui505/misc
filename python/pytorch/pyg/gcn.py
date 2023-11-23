#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2022 Institut Pasteur                                       #
#############################################################################

import os

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class GCN(torch.nn.Module):
    """
    See: https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs
    edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
    x: Node feature matrix with shape [num_nodes, num_node_features]
    edge_weight: Edge weight vector [num_edges]
    >>> edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    >>> x = torch.tensor([[0.3], [0.01], [0.5]], dtype=torch.float)
    >>> edge_weight = torch.tensor([0.1, 0.5, 0.2, 0.3], dtype=torch.float)
    >>> gcn = GCN(num_node_features=1, num_classes=4)
    >>> out = gcn(x, edge_index, edge_weight)
    >>> out.shape
    torch.Size([3, 4])

    """

    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)

        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    """
    See: https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs
    edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
    x: Node feature matrix with shape [num_nodes, num_node_features]
    edge_weight: Edge weight vector [num_edges]
    >>> edge_index = torch.tensor([[0, 1, 1, 2, 0, 1, 2], [1, 0, 2, 1, 0, 1, 2]], dtype=torch.long)
    >>> x = torch.tensor([[0.3], [0.01], [0.5], [0.3], [0.3], [0.3]], dtype=torch.float)
    >>> edge_weight = torch.tensor([0.1, 0.5, 0.2, 0.3], dtype=torch.float)
    >>> gat = GAT(num_node_features=1, num_classes=4)
    >>> out, outedge, outweight = gat(x, edge_index, edge_weight)
    >>> out.shape
    torch.Size([6, 4])
    >>> outedge
    tensor([[0, 1, 1, 2, 0, 1, 2],
            [1, 0, 2, 1, 0, 1, 2]])
    >>> outweight.shape
    torch.Size([7, 1])
    """

    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.conv1 = GATConv(num_node_features, 16, add_self_loops=False)
        self.conv2 = GATConv(16, num_classes, add_self_loops=False)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x, (out_edge_index,
            out_edge_weight) = self.conv2(x,
                                          edge_index,
                                          edge_weight,
                                          return_attention_weights=True)
        return F.log_softmax(x, dim=1), out_edge_index, out_edge_weight


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
    import argparse
    import doctest
    import sys

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
    parser.add_argument('--func',
                        help='Test only the given function(s)',
                        nargs='+')
    args = parser.parse_args()

    # If log is present log the arguments to the log file:
    for k, v in args._get_kwargs():
        log(f'# {k}: {v}')

    if args.test:
        if args.func is None:
            doctest.testmod(optionflags=doctest.ELLIPSIS
                            | doctest.REPORT_ONLY_FIRST_FAILURE)
        else:
            for f in args.func:
                print(f'Testing {f}')
                f = getattr(sys.modules[__name__], f)
                doctest.run_docstring_examples(f, globals())
        sys.exit()
