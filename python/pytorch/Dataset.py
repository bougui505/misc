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

import torch
import os


class Dataset(torch.utils.data.Dataset):
    """
    See: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

    Attributes:
        datapath_y
        list_IDs
        datapath_X

    >>> # Generate random datapoints for testing
    >>> dataset = Dataset('X_data', 'y_data', [0, 1, 2, 3])
    >>> dataset.generate_random()
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)
    >>> X, y = next(iter(dataloader))
    >>> X.shape
    torch.Size([2, 10, 10, 10])
    >>> y.shape
    torch.Size([2, 3, 3])

    """
    def __init__(self, datapath_X, datapath_y, list_IDs):
        self.list_IDs = list_IDs
        self.datapath_X = datapath_X
        self.datapath_y = datapath_y

    def generate_random(self):
        """
        Generate random points for testing purpose only

        """
        try:
            os.mkdir(self.datapath_X)
        except FileExistsError:
            pass
        try:
            os.mkdir(self.datapath_y)
        except FileExistsError:
            pass
        for ID in range(self.__len__()):
            X = torch.rand(size=(10, 10, 10))
            y = torch.rand(size=(3, 3))
            torch.save(X, f"{self.datapath_X}/{ID}.pt")
            torch.save(y, f"{self.datapath_y}/{ID}.pt")

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        X = torch.load(f'{self.datapath_X}/{ID}.pt')
        y = torch.load(f'{self.datapath_y}/{ID}.pt')
        return X, y


if __name__ == '__main__':
    import sys
    import argparse
    import doctest
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod()
        sys.exit()
