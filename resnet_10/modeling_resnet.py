#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------

import torch.nn as nn
from transformers import PreTrainedModel, ResNetConfig
from transformers.activations import ACT2FN


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation, stride=1, norm_groups=4):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels)
        self.act1 = ACT2FN[activation]
        self.act2 = ACT2FN[activation]
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if x.shape != out.shape:
            out += self.shortcut(x)
        else:
            out += x
        return self.act2(out)


class ResNet10(PreTrainedModel):
    config_class = ResNetConfig

    def __init__(self, config):
        super().__init__(config)

        self.embedder = nn.Sequential(
            nn.Conv2d(
                self.config.num_channels,
                self.config.embedding_size,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            # The original code has a small trick -
            # https://github.com/rail-berkeley/hil-serl/blob/main/serl_launcher/serl_launcher/vision/resnet_v1.py#L119
            # class MyGroupNorm(nn.GroupNorm):
            #     def __call__(self, x):
            #         if x.ndim == 3:
            #             x = x[jnp.newaxis]
            #             x = super().__call__(x)
            #             return x[0]
            #         else:
            #             return super().__call__(x)
            nn.GroupNorm(num_groups=4, eps=1e-5, num_channels=self.config.embedding_size),
            ACT2FN[self.config.hidden_act],
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.encoder = nn.Sequential()

        for i, size in enumerate(self.config.hidden_sizes):
            if i == 0:
                self.encoder.append(
                    BasicBlock(
                        self.config.embedding_size,
                        size,
                        activation=self.config.hidden_act,
                    )
                )
            else:
                self.encoder.append(
                    BasicBlock(
                        self.config.hidden_sizes[i - 1],
                        size,
                        activation=self.config.hidden_act,
                        stride=2,
                    )
                )

    def forward(self, x):
        out = self.embedder(x)
        out = self.encoder(out)

        return out
