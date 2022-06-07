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
from utils import Normalizer


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
               Linear-32                  [-1, 512]       2,097,664
    ================================================================
    Total params: 97,642,880
    Trainable params: 97,642,880
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.01
    Forward/backward pass size (MB): 17.35
    Params size (MB): 372.48
    Estimated Total Size (MB): 389.84
    ----------------------------------------------------------------

    >>> out = encoder(inp)
    >>> out.shape
    torch.Size([3, 512])
    """
    def __init__(self, latent_dims, input_size=(224, 224), interpolate=True, sample=True):
        super().__init__()
        self.input_size = input_size
        self.interpolate = interpolate
        self.sample = sample
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
            # x = utils.resize(x, size=self.input_size)  # perform padding or interpolation
        out = self.layers(x)
        mu = self.linear_mu(out)
        log_sigma_sq = self.linear_sigma(out)
        sigma_sq = torch.exp(log_sigma_sq)
        if self.sample:
            z = mu + torch.sqrt(sigma_sq) * self.N.sample(mu.shape)
        else:
            z = mu
        # See: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
        self.kl.append(((1 / 2) * (sigma_sq + mu**2 - 1 - log_sigma_sq).sum(axis=1)).mean())
        return z


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


def plot_latent(A, encoder):
    normalizer = Normalizer(A)
    A = normalizer.transform(normalizer.batch)[0][None, ...]
    z = encoder(A)
    z = z.detach().cpu().numpy()
    z = np.squeeze(z)
    # z = z / z.max(axis=-1)
    # z = (z - z.mean(axis=-1)) / z.std(axis=-1)
    z = z / np.linalg.norm(z, axis=-1)
    plt.plot(z[:40])
    return z


if __name__ == '__main__':
    import sys
    import doctest
    import argparse
    import numpy as np
    import matplotlib.pyplot as plt
    from misc.protein.VAE import cmapvae, PDBloader
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

    # A = torch.randn((1, 1, 78, 78))
    pdbdataset = PDBloader.PDBdataset(pdblist=['/home/bougui/source/misc/python/protein/VAE/data/1ycr.pdb'],
                                      selection='polymer.protein and name CA and chain A',
                                      interpolate=False)
    A = pdbdataset.__getitem__(0)
    nref = A.shape[-1]
    model = cmapvae.load_model(filename='/home/bougui/source/misc/python/protein/VAE/models/cmapvae_20220525_0843.pt',
                               latent_dims=512)
    # encoder = Encoder(512, sample=False)
    encoder = model.encoder
    pdbdataset = PDBloader.PDBdataset(pdblist=['pdb/pn/pdb6pno.ent.gz'],
                                      selection='polymer.protein and name CA and chain A',
                                      interpolate=False)
    A_rand = pdbdataset.__getitem__(0)
    z_rand = plot_latent(A_rand, encoder)
    z_ori = plot_latent(A, encoder)
    sim_rand = z_rand.dot(z_ori)
    dist_rand = np.linalg.norm(z_rand - z_ori)
    for i in range(1, 20):
        z = plot_latent(A[..., i:-i, i:-i], encoder)
        sim = z.dot(z_ori)
        dist = np.linalg.norm(z - z_ori)
        print(i, sim, dist, sim_rand, dist_rand)
    plt.show()
