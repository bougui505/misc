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
from functools import partial
from typing import Callable, List, Sequence, Type, Union

import torch
import torch.nn as nn
import torchvision.models.video.resnet as resnet
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torchvision.utils import _log_api_usage_once

# model = resnet.r3d_18(num_classes=400)
# model = resnet._video_resnet(block=resnet.BasicBlock,
#                              conv_makers=[resnet.Conv3DSimple] * 4,
#                              layers=[2, 2, 2, 2],
#                              stem=resnet.BasicStem,
#                              weights=None,
#                              progress=progress)


class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem"""

    def __init__(self, in_channels=3) -> None:
        super().__init__(
            nn.Conv3d(in_channels,
                      64,
                      kernel_size=(3, 7, 7),
                      stride=(1, 2, 2),
                      padding=(1, 3, 3),
                      bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )


def resnet3d(in_channels=3, out_channels=400):
    """
    >>> n_channel = 1
    >>> bs = 3
    >>> model = resnet3d(in_channels=n_channel, out_channels=512)
    >>> data = torch.randn(bs ,n_channel, 10, 11, 12)
    >>> out = model(data)
    >>> out.shape
    torch.Size([3, 512])
    """
    stem = partial(BasicStem, in_channels=in_channels)
    conv_makers = [resnet.Conv3DSimple] * 4
    model = VideoResNet(block=resnet.BasicBlock,
                        conv_makers=conv_makers,
                        layers=[2, 2, 2, 2],
                        stem=stem,
                        num_classes=out_channels)
    return model


class VideoResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[resnet.BasicBlock, resnet.Bottleneck]],
        conv_makers: Sequence[Type[Union[resnet.Conv3DSimple,
                                         resnet.Conv3DNoTemporal,
                                         resnet.Conv2Plus1D]]],
        layers: List[int],
        stem: Callable[..., nn.Module],
        num_classes: int = 400,
        zero_init_residual: bool = False,
    ) -> None:
        """Generic resnet video generator.

        Args:
            block (Type[Union[BasicBlock, Bottleneck]]): resnet building block
            conv_makers (List[Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]]]): generator
                function for each layer
            layers (List[int]): number of blocks per layer
            stem (Callable[..., nn.Module]): module specifying the ResNet stem.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super().__init__()
        _log_api_usage_once(self)
        self.inplanes = 64

        self.stem = stem()

        self.layer1 = self._make_layer(block,
                                       conv_makers[0],
                                       64,
                                       layers[0],
                                       stride=1)
        self.layer2 = self._make_layer(block,
                                       conv_makers[1],
                                       128,
                                       layers[1],
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       conv_makers[2],
                                       256,
                                       layers[2],
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       conv_makers[3],
                                       512,
                                       layers[3],
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, resnet.Bottleneck):
                    nn.init.constant_(m.bn3.weight,
                                      0)  # type: ignore[union-attr, arg-type]

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)

        # x = self.layer1(x)
        x = checkpoint(self.layer1, x)

        # x = self.layer2(x)
        x = checkpoint(self.layer2, x)

        # x = self.layer3(x)
        x = checkpoint(self.layer3, x)

        # x = self.layer4(x)
        x = checkpoint(self.layer4, x)

        x = self.avgpool(x)
        # Flatten the layer to fc
        x = x.flatten(1)

        # x = self.fc(x)
        x = checkpoint(self.fc, x)

        return x

    def _make_layer(
        self,
        block: Type[Union[resnet.BasicBlock, resnet.Bottleneck]],
        conv_builder: Type[Union[resnet.Conv3DSimple, resnet.Conv3DNoTemporal,
                                 resnet.Conv2Plus1D]],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=ds_stride,
                          bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)


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
        doctest.testmod(optionflags=doctest.ELLIPSIS
                        | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
