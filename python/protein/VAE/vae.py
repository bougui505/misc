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

# See: https://avandekleut.github.io/vae/

import torch


class Encoder(torch.nn.Module):
    """
    >>> batch = 3
    >>> inp = torch.ones(batch, 1, 50, 50)
    >>> encoder = Encoder(10)
    >>> out = encoder(inp)
    >>> out.shape
    torch.Size([3, 10])
    """
    def __init__(self, latent_dims, input_size=(224, 224), interpolate=True):
        super().__init__()
        self.input_size = input_size
        self.interpolate = interpolate
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4)
        self.conv2 = torch.nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=2)
        self.conv4 = torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding='same', stride=1)
        self.conv5 = torch.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=2)
        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(in_features=6400, out_features=4096)
        self.linear2 = torch.nn.Linear(in_features=4096, out_features=4096)
        self.linear_mu = torch.nn.Linear(in_features=4096, out_features=latent_dims)
        self.linear_sigma = torch.nn.Linear(in_features=4096, out_features=latent_dims)
        self.layers = torch.nn.Sequential(self.conv1, self.relu, self.conv2, self.relu, self.conv3, self.relu,
                                          self.conv4, self.relu, self.conv5, self.relu, self.flatten, self.linear1,
                                          self.relu, self.linear2, self.relu)
        self.N = torch.distributions.Normal(0, 1)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()
        self.reset_kl()

    def reset_kl(self):
        self.kl = []

    def forward(self, x):
        if self.interpolate:
            x = torch.nn.functional.interpolate(x, size=self.input_size)
        out = self.layers(x)
        mu = self.linear_mu(out)
        log_sigma_sq = self.linear_sigma(out)
        sigma_sq = torch.exp(log_sigma_sq)
        z = mu + torch.sqrt(sigma_sq) * self.N.sample(mu.shape)
        # See: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
        self.kl.append(((1 / 2) * (sigma_sq + mu**2 - 1 - log_sigma_sq).sum(axis=1)).mean())
        return z


class Decoder(torch.nn.Module):
    def __init__(self, latent_dims, interpolate=True):
        """
        >>> batch = 3
        >>> latent_dims = 10
        >>> z = torch.randn(batch, latent_dims)
        >>> decoder = Decoder(latent_dims=latent_dims)
        >>> out = decoder(z, output_size=(50,50))
        >>> out.shape
        torch.Size([3, 1, 50, 50])
        """
        super().__init__()
        self.interpolate = interpolate
        self.linear1 = torch.nn.Linear(in_features=latent_dims, out_features=4096)
        self.linear2 = torch.nn.Linear(in_features=4096, out_features=4096)
        self.linear3 = torch.nn.Linear(in_features=4096, out_features=6400)
        self.upconv1 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=384, kernel_size=4, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding='same', stride=1)
        self.upconv3 = torch.nn.ConvTranspose2d(in_channels=384, out_channels=256, kernel_size=4, stride=2)
        self.upconv4 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=96, kernel_size=4, stride=2)
        self.upconv5 = torch.nn.ConvTranspose2d(in_channels=96, out_channels=1, kernel_size=12, stride=4)
        self.relu = torch.nn.ReLU()

    def forward(self, z, output_size):
        B = z.shape[0]  # batch size
        out = self.linear1(z)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = torch.reshape(out, (B, 256, 5, 5))
        out = self.upconv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.upconv3(out)
        out = self.relu(out)
        out = self.upconv4(out)
        out = self.relu(out)
        out = self.upconv5(out)  # torch.Size([3, 1, 224, 224])
        # out = self.relu(out)
        if self.interpolate:
            out = torch.nn.functional.interpolate(out, size=output_size)
        return out


class VariationalAutoencoder(torch.nn.Module):
    """
    >>> batch = 3
    >>> inp = torch.ones(batch, 1, 50, 50)
    >>> vae = VariationalAutoencoder(10)
    >>> out = vae(inp)
    >>> out.shape
    torch.Size([3, 1, 50, 50])
    """
    def __init__(self, latent_dims, interpolate=True):
        super().__init__()
        self.encoder = Encoder(latent_dims, interpolate=interpolate)
        self.decoder = Decoder(latent_dims, interpolate=interpolate)

    def forward(self, x):
        output_size = x.shape[-2:]
        z = self.encoder(x)
        return self.decoder(z, output_size=output_size)


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
