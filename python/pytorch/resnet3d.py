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
import torchvision.models.video.resnet as resnet
from torchsummary import summary
from functools import partial
import torch.nn as nn
from typing import Type, Union, Sequence, List, Callable
from torchvision.utils import _log_api_usage_once
from torch import Tensor
from torch.utils.checkpoint import checkpoint

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
            nn.Conv3d(in_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )


def resnet3d(in_channels=3, out_channels=400):
    """
    >>> model = resnet3d()
    >>> summary(model, (3, 64, 64, 64))
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv3d-1       [-1, 64, 64, 32, 32]          28,224
           BatchNorm3d-2       [-1, 64, 64, 32, 32]             128
                  ReLU-3       [-1, 64, 64, 32, 32]               0
          Conv3DSimple-4       [-1, 64, 64, 32, 32]         110,592
           BatchNorm3d-5       [-1, 64, 64, 32, 32]             128
                  ReLU-6       [-1, 64, 64, 32, 32]               0
          Conv3DSimple-7       [-1, 64, 64, 32, 32]         110,592
           BatchNorm3d-8       [-1, 64, 64, 32, 32]             128
                  ReLU-9       [-1, 64, 64, 32, 32]               0
           BasicBlock-10       [-1, 64, 64, 32, 32]               0
         Conv3DSimple-11       [-1, 64, 64, 32, 32]         110,592
          BatchNorm3d-12       [-1, 64, 64, 32, 32]             128
                 ReLU-13       [-1, 64, 64, 32, 32]               0
         Conv3DSimple-14       [-1, 64, 64, 32, 32]         110,592
          BatchNorm3d-15       [-1, 64, 64, 32, 32]             128
                 ReLU-16       [-1, 64, 64, 32, 32]               0
           BasicBlock-17       [-1, 64, 64, 32, 32]               0
         Conv3DSimple-18      [-1, 128, 32, 16, 16]         221,184
          BatchNorm3d-19      [-1, 128, 32, 16, 16]             256
                 ReLU-20      [-1, 128, 32, 16, 16]               0
         Conv3DSimple-21      [-1, 128, 32, 16, 16]         442,368
          BatchNorm3d-22      [-1, 128, 32, 16, 16]             256
               Conv3d-23      [-1, 128, 32, 16, 16]           8,192
          BatchNorm3d-24      [-1, 128, 32, 16, 16]             256
                 ReLU-25      [-1, 128, 32, 16, 16]               0
           BasicBlock-26      [-1, 128, 32, 16, 16]               0
         Conv3DSimple-27      [-1, 128, 32, 16, 16]         442,368
          BatchNorm3d-28      [-1, 128, 32, 16, 16]             256
                 ReLU-29      [-1, 128, 32, 16, 16]               0
         Conv3DSimple-30      [-1, 128, 32, 16, 16]         442,368
          BatchNorm3d-31      [-1, 128, 32, 16, 16]             256
                 ReLU-32      [-1, 128, 32, 16, 16]               0
           BasicBlock-33      [-1, 128, 32, 16, 16]               0
         Conv3DSimple-34        [-1, 256, 16, 8, 8]         884,736
          BatchNorm3d-35        [-1, 256, 16, 8, 8]             512
                 ReLU-36        [-1, 256, 16, 8, 8]               0
         Conv3DSimple-37        [-1, 256, 16, 8, 8]       1,769,472
          BatchNorm3d-38        [-1, 256, 16, 8, 8]             512
               Conv3d-39        [-1, 256, 16, 8, 8]          32,768
          BatchNorm3d-40        [-1, 256, 16, 8, 8]             512
                 ReLU-41        [-1, 256, 16, 8, 8]               0
           BasicBlock-42        [-1, 256, 16, 8, 8]               0
         Conv3DSimple-43        [-1, 256, 16, 8, 8]       1,769,472
          BatchNorm3d-44        [-1, 256, 16, 8, 8]             512
                 ReLU-45        [-1, 256, 16, 8, 8]               0
         Conv3DSimple-46        [-1, 256, 16, 8, 8]       1,769,472
          BatchNorm3d-47        [-1, 256, 16, 8, 8]             512
                 ReLU-48        [-1, 256, 16, 8, 8]               0
           BasicBlock-49        [-1, 256, 16, 8, 8]               0
         Conv3DSimple-50         [-1, 512, 8, 4, 4]       3,538,944
          BatchNorm3d-51         [-1, 512, 8, 4, 4]           1,024
                 ReLU-52         [-1, 512, 8, 4, 4]               0
         Conv3DSimple-53         [-1, 512, 8, 4, 4]       7,077,888
          BatchNorm3d-54         [-1, 512, 8, 4, 4]           1,024
               Conv3d-55         [-1, 512, 8, 4, 4]         131,072
          BatchNorm3d-56         [-1, 512, 8, 4, 4]           1,024
                 ReLU-57         [-1, 512, 8, 4, 4]               0
           BasicBlock-58         [-1, 512, 8, 4, 4]               0
         Conv3DSimple-59         [-1, 512, 8, 4, 4]       7,077,888
          BatchNorm3d-60         [-1, 512, 8, 4, 4]           1,024
                 ReLU-61         [-1, 512, 8, 4, 4]               0
         Conv3DSimple-62         [-1, 512, 8, 4, 4]       7,077,888
          BatchNorm3d-63         [-1, 512, 8, 4, 4]           1,024
                 ReLU-64         [-1, 512, 8, 4, 4]               0
           BasicBlock-65         [-1, 512, 8, 4, 4]               0
    AdaptiveAvgPool3d-66         [-1, 512, 1, 1, 1]               0
               Linear-67                  [-1, 400]         205,200
    ================================================================
    Total params: 33,371,472
    Trainable params: 33,371,472
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 3.00
    Forward/backward pass size (MB): 712.01
    Params size (MB): 127.30
    Estimated Total Size (MB): 842.31
    ----------------------------------------------------------------
    >>> model = resnet3d(in_channels=1, out_channels=256)
    >>> summary(model, (1, 64, 64, 64))
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv3d-1       [-1, 64, 64, 32, 32]           9,408
           BatchNorm3d-2       [-1, 64, 64, 32, 32]             128
                  ReLU-3       [-1, 64, 64, 32, 32]               0
          Conv3DSimple-4       [-1, 64, 64, 32, 32]         110,592
           BatchNorm3d-5       [-1, 64, 64, 32, 32]             128
                  ReLU-6       [-1, 64, 64, 32, 32]               0
          Conv3DSimple-7       [-1, 64, 64, 32, 32]         110,592
           BatchNorm3d-8       [-1, 64, 64, 32, 32]             128
                  ReLU-9       [-1, 64, 64, 32, 32]               0
           BasicBlock-10       [-1, 64, 64, 32, 32]               0
         Conv3DSimple-11       [-1, 64, 64, 32, 32]         110,592
          BatchNorm3d-12       [-1, 64, 64, 32, 32]             128
                 ReLU-13       [-1, 64, 64, 32, 32]               0
         Conv3DSimple-14       [-1, 64, 64, 32, 32]         110,592
          BatchNorm3d-15       [-1, 64, 64, 32, 32]             128
                 ReLU-16       [-1, 64, 64, 32, 32]               0
           BasicBlock-17       [-1, 64, 64, 32, 32]               0
         Conv3DSimple-18      [-1, 128, 32, 16, 16]         221,184
          BatchNorm3d-19      [-1, 128, 32, 16, 16]             256
                 ReLU-20      [-1, 128, 32, 16, 16]               0
         Conv3DSimple-21      [-1, 128, 32, 16, 16]         442,368
          BatchNorm3d-22      [-1, 128, 32, 16, 16]             256
               Conv3d-23      [-1, 128, 32, 16, 16]           8,192
          BatchNorm3d-24      [-1, 128, 32, 16, 16]             256
                 ReLU-25      [-1, 128, 32, 16, 16]               0
           BasicBlock-26      [-1, 128, 32, 16, 16]               0
         Conv3DSimple-27      [-1, 128, 32, 16, 16]         442,368
          BatchNorm3d-28      [-1, 128, 32, 16, 16]             256
                 ReLU-29      [-1, 128, 32, 16, 16]               0
         Conv3DSimple-30      [-1, 128, 32, 16, 16]         442,368
          BatchNorm3d-31      [-1, 128, 32, 16, 16]             256
                 ReLU-32      [-1, 128, 32, 16, 16]               0
           BasicBlock-33      [-1, 128, 32, 16, 16]               0
         Conv3DSimple-34        [-1, 256, 16, 8, 8]         884,736
          BatchNorm3d-35        [-1, 256, 16, 8, 8]             512
                 ReLU-36        [-1, 256, 16, 8, 8]               0
         Conv3DSimple-37        [-1, 256, 16, 8, 8]       1,769,472
          BatchNorm3d-38        [-1, 256, 16, 8, 8]             512
               Conv3d-39        [-1, 256, 16, 8, 8]          32,768
          BatchNorm3d-40        [-1, 256, 16, 8, 8]             512
                 ReLU-41        [-1, 256, 16, 8, 8]               0
           BasicBlock-42        [-1, 256, 16, 8, 8]               0
         Conv3DSimple-43        [-1, 256, 16, 8, 8]       1,769,472
          BatchNorm3d-44        [-1, 256, 16, 8, 8]             512
                 ReLU-45        [-1, 256, 16, 8, 8]               0
         Conv3DSimple-46        [-1, 256, 16, 8, 8]       1,769,472
          BatchNorm3d-47        [-1, 256, 16, 8, 8]             512
                 ReLU-48        [-1, 256, 16, 8, 8]               0
           BasicBlock-49        [-1, 256, 16, 8, 8]               0
         Conv3DSimple-50         [-1, 512, 8, 4, 4]       3,538,944
          BatchNorm3d-51         [-1, 512, 8, 4, 4]           1,024
                 ReLU-52         [-1, 512, 8, 4, 4]               0
         Conv3DSimple-53         [-1, 512, 8, 4, 4]       7,077,888
          BatchNorm3d-54         [-1, 512, 8, 4, 4]           1,024
               Conv3d-55         [-1, 512, 8, 4, 4]         131,072
          BatchNorm3d-56         [-1, 512, 8, 4, 4]           1,024
                 ReLU-57         [-1, 512, 8, 4, 4]               0
           BasicBlock-58         [-1, 512, 8, 4, 4]               0
         Conv3DSimple-59         [-1, 512, 8, 4, 4]       7,077,888
          BatchNorm3d-60         [-1, 512, 8, 4, 4]           1,024
                 ReLU-61         [-1, 512, 8, 4, 4]               0
         Conv3DSimple-62         [-1, 512, 8, 4, 4]       7,077,888
          BatchNorm3d-63         [-1, 512, 8, 4, 4]           1,024
                 ReLU-64         [-1, 512, 8, 4, 4]               0
           BasicBlock-65         [-1, 512, 8, 4, 4]               0
    AdaptiveAvgPool3d-66         [-1, 512, 1, 1, 1]               0
               Linear-67                  [-1, 256]         131,328
    ================================================================
    Total params: 33,278,784
    Trainable params: 33,278,784
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 1.00
    Forward/backward pass size (MB): 712.01
    Params size (MB): 126.95
    Estimated Total Size (MB): 839.95
    ----------------------------------------------------------------
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
        conv_makers: Sequence[Type[Union[resnet.Conv3DSimple, resnet.Conv3DNoTemporal, resnet.Conv2Plus1D]]],
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

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[union-attr, arg-type]

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
        conv_builder: Type[Union[resnet.Conv3DSimple, resnet.Conv3DNoTemporal, resnet.Conv2Plus1D]],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

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