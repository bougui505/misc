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
import torch.nn as nn
from misc.eta import ETA
import time
import datetime
from torchsummary import summary
from functools import partial
import numpy as np
from torch.utils.checkpoint import checkpoint_sequential
from termcolor import colored
# ### UNCOMMENT FOR LOGGING ####
import logging

logfilename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
logging.info(f"################ Starting {__file__} ################")
# ### ##################### ####


def save_model(model, filename):
    torch.save(model.state_dict(), filename)


def load_model(model, filename):
    """
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load(filename, map_location=torch.device(device)))
    model.eval()
    return model


class Dataset_test(torch.utils.data.Dataset):
    """
    >>> dataset = Dataset_test(ndata=10, shape=(3, 64, 64), nclasses=10)
    >>> X, y = dataset[0]
    >>> X.shape
    torch.Size([3, 64, 64])
    >>> y.shape
    torch.Size([])
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)
    >>> X, y = next(iter(dataloader))
    >>> X.shape
    torch.Size([2, 3, 64, 64])
    >>> y.shape
    torch.Size([2])
    """
    def __init__(self, ndata, shape, nclasses):
        self.ids = range(ndata)
        self.shape = shape
        self.nclasses = nclasses

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        X = torch.randn(self.shape)
        y = torch.randint(low=0, high=self.nclasses, size=(1, ))[0]
        return X, y


class Model_test(nn.Sequential):
    """
    >>> seed = torch.manual_seed(0)
    >>> model = Model_test()
    >>> summary(model, (3, 64, 64))
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1            [-1, 5, 64, 64]             135
           BatchNorm2d-2            [-1, 5, 64, 64]              10
                  ReLU-3            [-1, 5, 64, 64]               0
               Flatten-4                [-1, 20480]               0
                Linear-5                   [-1, 10]         204,810
    ================================================================
    Total params: 204,955
    Trainable params: 204,955
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.05
    Forward/backward pass size (MB): 0.63
    Params size (MB): 0.78
    Estimated Total Size (MB): 1.45
    ----------------------------------------------------------------
    >>> dataset = Dataset_test(ndata=10, shape=(3, 64, 64), nclasses=10)
    >>> X, y = dataset[0]
    >>> X.shape
    torch.Size([3, 64, 64])
    >>> y.shape
    torch.Size([])
    >>> out = model(X[None, ...])
    >>> out.shape
    torch.Size([1, 10])
    >>> loss = nn.CrossEntropyLoss()
    >>> loss(out, y[None, ...])
    tensor(2.8418, grad_fn=<NllLossBackward0>)
    """
    def __init__(self, in_channels=3, nclasses=10):
        super().__init__(nn.Conv2d(in_channels, 5, kernel_size=(3, 3), stride=1, padding='same', bias=False),
                         nn.BatchNorm2d(5), nn.ReLU(inplace=True), nn.Flatten(), nn.Linear(20480, nclasses))


def sequential_checkpointer(model, segments):
    """
    Create a checkpointed forward function. Could be used when tensors are larger than GPU memory.
    See: https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
    - segments: set the number of checkpoint segments

    >>> model = Model_test()
    >>> forward_batch = sequential_checkpointer(model, 2)
    >>> dataset = Dataset_test(ndata=10, shape=(3, 64, 64), nclasses=10)
    >>> X, y = dataset[0]
    >>> X.shape
    torch.Size([3, 64, 64])
    >>> y.shape
    torch.Size([])
    >>> X = torch.autograd.Variable(X[None, ...], requires_grad=True)
    >>> out = forward_batch(X, model)
    >>> out.shape
    torch.Size([1, 10])
    """
    modules = [module for k, module in model._modules.items()]

    # out = checkpoint_sequential(modules, segments, input_var)
    # forward_batch = partial(checkpoint_sequential, functions=modules, segments=segments)
    def forward_batch(X, model):
        """
        X: input to forward
        model: argument is there just for compatibility with train
        """
        return checkpoint_sequential(functions=modules, segments=segments, input=X)

    return forward_batch


def train(model,
          loss_function,
          dataloader,
          n_epochs,
          forward_batch,
          modelfilename='model.pt',
          save_each=30,
          print_each=100,
          save_each_epoch=True,
          early_break=np.inf):
    """
    - save_each: save model every the given number of minutes

    >>> seed = torch.manual_seed(0)
    >>> model = Model_test()
    >>> loss = nn.CrossEntropyLoss()
    >>> dataset = Dataset_test(ndata=10, shape=(3, 64, 64), nclasses=3)
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)
    >>> forward_batch = lambda batch, model: model(batch[0])
    >>> loss_function = lambda batch, out: loss(out, batch[1])
    >>> X, y = dataset[0]
    >>> batch = (X[None, ...], y[None, ...])
    >>> out = forward_batch(batch, model)
    >>> out.shape
    torch.Size([1, 10])
    >>> loss_function(batch, out)
    tensor(2.5691, grad_fn=<NllLossBackward0>)

    >>> train(model, loss_function, dataloader, 10, forward_batch, print_each=1)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if os.path.exists(modelfilename):
        log(f'# Loading model from {modelfilename}')
        model = load_model(model, filename=modelfilename)
        model.train()
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters())
    save_model(model, modelfilename)
    epoch = 0
    step = 0
    total_steps = n_epochs * len(dataloader)
    t_0 = time.time()
    eta = ETA(total_steps=total_steps)
    for epoch in range(n_epochs):
        for batch in dataloader:
            step += 1
            try:
                out = forward_batch(batch, model)
                loss_val = loss_function(batch, out)
                if device == 'cuda':
                    memusage = torch.cuda.memory_allocated() * 100 / torch.cuda.max_memory_allocated()
                else:
                    memusage = 0.
                loss_val.backward()
                opt.step()
            except (RuntimeError, ValueError) as error:
                memusage = 111
                outstr = f'WARNING: forward error for batch at step: {step}\nERROR: {error}'
                outstr = colored(outstr, 'red')
                print(outstr)
            opt.zero_grad()
            if (time.time() - t_0) / 60 >= save_each:
                t_0 = time.time()
                save_model(model, modelfilename)
            if not step % print_each:
                eta_val = eta(step)
                last_saved = (time.time() - t_0)
                last_saved = str(datetime.timedelta(seconds=last_saved))
                log(f"epoch: {epoch+1}|step: {step}/{total_steps}|loss: {loss_val}|last_saved: {last_saved}|gpu_memory_usage: {memusage:.2f}%|eta: {eta_val}"
                    )
            if step >= early_break:
                break
        if save_each_epoch:
            t_0 = time.time()
            save_model(model, modelfilename)


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
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-a', '--arg1')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
