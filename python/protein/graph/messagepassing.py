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
import torch
from torch_geometric.nn import MessagePassing, Sequential
from torch_geometric.nn.models import GAT


class Graph_conv(MessagePassing):
    """
    >>> import proteingraph
    >>> node_features, edge_index, edge_features = proteingraph.prot_to_graph('data/1t4e.pdb')
    >>> node_features.shape
    torch.Size([784, 58])
    >>> edge_index.shape
    torch.Size([2, 18610])
    >>> edge_features.shape
    torch.Size([18610, 1])
    >>> n_n = node_features.shape[1]
    >>> n_e = edge_features.shape[1]
    >>> graph_conv = Graph_conv(n_n, n_e, 512)
    >>> out = graph_conv(node_features, edge_index, edge_features)
    >>> out.shape
    torch.Size([784, 512])

    >>> count_parameters(graph_conv)
    293888
    """

    def __init__(self, n_n, n_e, n_o):
        """
        n_n: number of node features
        n_e: number of edge features
        n_o: number of output features
        """
        super().__init__(aggr="add")
        self.lin_nodes = torch.nn.Linear(n_n, n_o)
        self.lin_edges = torch.nn.Linear(n_e, n_o)
        self.lin_message = torch.nn.Linear(n_o, n_o)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_nodes.reset_parameters()
        self.lin_edges.reset_parameters()
        self.lin_message.reset_parameters()

    def forward(self, x, edge_index, edge_features):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        out = self.propagate(edge_index, x=x, edge_features=edge_features)
        return out

    def message(self, x_j, edge_features):
        m_n = self.lin_nodes(x_j)
        m_e = self.lin_edges(edge_features)
        m = torch.tanh(self.lin_message(m_n * m_e))
        return m

    def update(self, aggr_out, x):
        out = self.lin_nodes(x) + aggr_out
        return out


class GCN(torch.nn.Module):
    """
    >>> import proteingraph
    >>> node_features, edge_index, edge_features = proteingraph.prot_to_graph('data/1t4e.pdb')
    >>> node_features.shape
    torch.Size([784, 58])
    >>> edge_index.shape
    torch.Size([2, 18610])
    >>> edge_features.shape
    torch.Size([18610, 1])
    >>> n_n = node_features.shape[1]
    >>> n_e = edge_features.shape[1]
    >>> gcn = GCN(n_n, n_e, n_o=256, embedding_dim=512)
    >>> out = gcn(node_features, edge_index, edge_features)
    >>> out.shape
    torch.Size([512])

    # >>> count_parameters(gcn)
    # 3758650

    Try a dataloader:
    >>> dataset = proteingraph.Dataset('data/dude_test_100.smi', return_pyg_graph=True)
    >>> from torch_geometric.loader import DataLoader
    >>> loader = DataLoader(dataset, batch_size=8)
    >>> for batch in loader:
    ...     break
    >>> batch
    DataBatch(x=[1710, 58], edge_index=[2, 28294], edge_attr=[28294, 1], y=[8], batch=[1710], ptr=[9])
    >>> batch.batch
    tensor([0, 0, 0,  ..., 7, 7, 7])

    >>> gcn = GCN(n_n, n_e, n_o=256, embedding_dim=512)
    >>> out = gcn(batch.x, batch.edge_index, batch.edge_attr, batch_index=batch.batch)
    >>> out.shape
    torch.Size([8, 512])

    Return node features
    >>> out = gcn(batch.x, batch.edge_index, batch.edge_attr, batch_index=batch.batch, return_node_features=True)
    >>> len(out)
    8
    >>> [e.shape for e in out]
    [torch.Size([227, 58]), torch.Size([227, 58]), torch.Size([249, 58]), torch.Size([192, 58]), torch.Size([258, 58]), torch.Size([200, 58]), torch.Size([168, 58]), torch.Size([189, 58])]

    Check with attention
    >>> gcn = GCN(n_n, n_e, n_o=256, embedding_dim=512, attention=True)
    >>> out = gcn(batch.x, batch.edge_index, batch.edge_attr, batch_index=batch.batch)
    >>> out.shape
    torch.Size([8, 512])


    """

    def __init__(
        self, n_n, n_e, n_o, embedding_dim, nlayers=28, normalize=False, attention=False
    ):
        """
        n_n: number of node features
        n_e: number of edge features
        n_o: number of output features
        normalize: If normalize, enforce the output vector to unit vector
        """
        super().__init__()
        layers = [(Graph_conv(n_n, n_e, n_o), "x, edge_index, edge_attr -> x")]
        if not attention:
            for _ in range(nlayers - 1):
                layers.append(
                    (Graph_conv(n_o, n_e, n_o), "x, edge_index, edge_attr -> x")
                )
            self.convolutions = Sequential("x, edge_index, edge_attr", layers)
        else:
            self.convolutions = GAT(
                in_channels=n_n, hidden_channels=n_o, num_layers=nlayers, v2=False
            )
        self.linear = torch.nn.Linear(n_o, embedding_dim)
        self.linear_node_feature = torch.nn.Linear(embedding_dim, n_n)
        self.normalize = normalize

    def forward(
        self, x, edge_index, edge_features, batch_index=None, return_node_features=False
    ):
        out = self.convolutions(x, edge_index, edge_attr=edge_features)
        out = self.linear(out)
        if batch_index is not None:
            labels = torch.unique(batch_index)
            if not return_node_features:
                out = torch.stack([out[batch_index == i].mean(dim=-2) for i in labels])
            else:
                out = self.linear_node_feature(out)
                out = [out[batch_index == i] for i in labels]
        else:
            if not return_node_features:
                out = torch.mean(out, dim=-2)
            else:
                out = self.linear_node_feature(out)
            # out = self.linear(out)
        if self.normalize:
            out = torch.nn.functional.normalize(out, dim=-1)
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    # import logging
    # if not os.path.isdir('logs'):
    #     os.mkdir('logs')
    # logfilename = 'logs/' + os.path.splitext(os.path.basename(__file__))[0] + '.log'
    # logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    # logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument("-a", "--arg1")
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
