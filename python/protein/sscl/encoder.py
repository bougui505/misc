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

import torch
from torchsummary import summary
from misc.protein.sscl.utils import normalize


class Encoder(torch.nn.Module):
    """
    >>> batch = 3
    >>> inp = torch.ones(batch, 1, 50, 50)
    >>> encoder = Encoder(512)
    >>> summary(encoder, (1, 50, 50))
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 96, 54, 54]          11,712
                Conv2d-2           [-1, 96, 54, 54]          11,712
                  ReLU-3           [-1, 96, 54, 54]               0
                  ReLU-4           [-1, 96, 54, 54]               0
                Conv2d-5          [-1, 256, 25, 25]         614,656
                Conv2d-6          [-1, 256, 25, 25]         614,656
                  ReLU-7          [-1, 256, 25, 25]               0
                  ReLU-8          [-1, 256, 25, 25]               0
                Conv2d-9          [-1, 384, 12, 12]         885,120
               Conv2d-10          [-1, 384, 12, 12]         885,120
                 ReLU-11          [-1, 384, 12, 12]               0
                 ReLU-12          [-1, 384, 12, 12]               0
               Conv2d-13          [-1, 384, 12, 12]       1,327,488
               Conv2d-14          [-1, 384, 12, 12]       1,327,488
                 ReLU-15          [-1, 384, 12, 12]               0
                 ReLU-16          [-1, 384, 12, 12]               0
               Conv2d-17            [-1, 256, 5, 5]         884,992
               Conv2d-18            [-1, 256, 5, 5]         884,992
                 ReLU-19            [-1, 256, 5, 5]               0
                 ReLU-20            [-1, 256, 5, 5]               0
              Flatten-21                 [-1, 6400]               0
              Flatten-22                 [-1, 6400]               0
               Linear-23                 [-1, 4096]      26,218,496
               Linear-24                 [-1, 4096]      26,218,496
                 ReLU-25                 [-1, 4096]               0
                 ReLU-26                 [-1, 4096]               0
               Linear-27                 [-1, 4096]      16,781,312
               Linear-28                 [-1, 4096]      16,781,312
                 ReLU-29                 [-1, 4096]               0
                 ReLU-30                 [-1, 4096]               0
               Linear-31                  [-1, 512]       2,097,664
    ================================================================
    Total params: 95,545,216
    Trainable params: 95,545,216
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.01
    Forward/backward pass size (MB): 17.35
    Params size (MB): 364.48
    Estimated Total Size (MB): 381.83
    ----------------------------------------------------------------

    >>> out = encoder(inp)
    >>> out.shape
    torch.Size([3, 512])
    """
    def __init__(self, latent_dims, input_size=(224, 224), interpolate=True, normalize=True):
        super().__init__()
        self.input_size = input_size
        self.interpolate = interpolate
        self.normalize = normalize
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4)
        self.conv2 = torch.nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=2)
        self.conv4 = torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding='same', stride=1)
        self.conv5 = torch.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=2)
        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.LazyLinear(out_features=4096)
        self.linear2 = torch.nn.Linear(in_features=4096, out_features=4096)
        self.linear_mu = torch.nn.Linear(in_features=4096, out_features=latent_dims)
        self.layers = torch.nn.Sequential(self.conv1, self.relu, self.conv2, self.relu, self.conv3, self.relu,
                                          self.conv4, self.relu, self.conv5, self.relu, self.flatten, self.linear1,
                                          self.relu, self.linear2, self.relu)
        self.N = torch.distributions.Normal(0, 1)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()

    def forward(self, x):
        if self.normalize:
            x = normalize(x)
        if self.interpolate:
            x = torch.nn.functional.interpolate(x, size=self.input_size)
            # x = utils.resize(x, size=self.input_size)  # perform padding or interpolation
        # assert not torch.isnan(x).any(), 'Error: nan detected in network input'
        if torch.isnan(x).any():
            print('WARNING: nan detected in network input')
        out = self.layers(x)
        # assert not torch.isnan(out).any(), 'Error: nan detected in network output'
        if torch.isnan(out).any():
            print('WARNING: nan detected in network output')
        z = self.linear_mu(out)
        return z


def load_model(filename, latent_dims=512):
    """
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Encoder(latent_dims=latent_dims)
    model.load_state_dict(torch.load(filename, map_location=torch.device(device)))
    model.eval()
    return model


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


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
