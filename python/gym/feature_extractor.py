#!/usr/bin/env python3
# -*- coding: UTF8 -*-
import sys

import gym
import torch as torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self,
                 observation_space: gym.spaces.Dict,
                 features_dim: int = 32,
                 history=1,
                 discrete_action_space=False,
                 include_traj=False,
                 include_move=False
                 ):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        self.history = history
        self.discrete_action_space = discrete_action_space,
        self.include_traj = include_traj
        self.include_move = include_move

        sample = observation_space.sample()
        sample_img = sample['values']
        sample_img = sample_img[None, ...]
        n_input_channels = 1
        if include_traj:
            sample_past = sample['past_img'][None, ...]
            n_input_channels = 2
        self.cnn = nn.ModuleList((
            nn.Conv3d(n_input_channels, 8, kernel_size=(history, 4, 4), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(8, 8, kernel_size=(1, 4, 4), stride=1, padding=0),
            nn.ReLU(),
            # nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(8, 16, kernel_size=(1, 3, 3), stride=1, padding=0),
            nn.Flatten()))

        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample_input = torch.as_tensor(sample_img)
            if include_traj:
                sample_input_past = torch.as_tensor(sample_past)
                sample_input = torch.cat((sample_input, sample_input_past), dim=1)
            n_flatten = self.forward_cnn(sample_input).float().shape[1]
        n_flatten += 2 * self.history * self.include_move
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward_cnn(self, input):
        for layer in self.cnn:
            input = layer(input)
            # print(input.shape)
        return input

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Get inputs and concatenate
        input_tensor = observations['values']
        if self.include_traj:
            past = observations['past_img']
            input_tensor = torch.cat((input_tensor, past), dim=1)

        # Feed to CNN and linear
        total_feats = self.forward_cnn(input_tensor)
        if self.include_move:
            move = observations['movement']
            pos_feats = move.flatten(start_dim=1)
            total_feats = torch.cat((total_feats, pos_feats), dim=1)
        preds = self.linear(total_feats)
        return preds


class CustomCNN2D(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self,
                 observation_space: gym.spaces.Dict,
                 # observation_space: gym.spaces.Box,
                 features_dim: int = 32,
                 history=1):
        super(CustomCNN2D, self).__init__(observation_space, features_dim)

        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # We get a sample input for shapes. Then images are processed through a CNN that yields a single vector

        sample = observation_space.sample()
        sample_img = sample['values']
        sample_img = sample_img[None, ...]
        sample_pos = sample['movement']

        n_input_channels = sample_img.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 8, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0),
            nn.Flatten())

        # Compute shape by doing one forward pass
        with torch.no_grad():
            self.cnn(torch.as_tensor(sample_img))
            n_flatten = self.cnn(torch.as_tensor(sample_img).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten + 2, features_dim),
                                    nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        image = observations['values']
        pos = observations['movement']
        img_feats = self.cnn(image)
        total_feats = torch.cat((img_feats, pos), dim=1)
        preds = self.linear(total_feats)
        return preds
