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

from typing import Optional

import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithNoAttention

from .configuration_resnet import ResNet10Config


class JaxStyleMaxPool(nn.Module):
    def forward(self, x):
        x = nn.functional.pad(x, (0, 1, 0, 1), value=-float('inf'))  # Pad right/bottom by 1 to match JAX's maxpooling padding="SAME"
        return nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)


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
        
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.shortcut is not None:
            identity = self.shortcut(identity)

        out += identity
        return self.act2(out)


class Encoder(nn.Module):
    def __init__(self, config: ResNet10Config):
        super().__init__()
        self.config = config
        self.stages = nn.ModuleList([])

        for i, size in enumerate(self.config.hidden_sizes):
            if i == 0:
                self.stages.append(
                    BasicBlock(
                        self.config.embedding_size,
                        size,
                        activation=self.config.hidden_act,
                    )
                )
            else:
                self.stages.append(
                    BasicBlock(
                        self.config.hidden_sizes[i - 1],
                        size,
                        activation=self.config.hidden_act,
                        stride=2,
                    )
                )

    def forward(self, hidden_state: Tensor, output_hidden_states: bool = False) -> BaseModelOutputWithNoAttention:
        hidden_states = () if output_hidden_states else None

        for stage in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)

            hidden_state = stage(hidden_state)

        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
        )


class ResNet10(PreTrainedModel):
    config_class = ResNet10Config

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
            JaxStyleMaxPool(),
        )

        self.encoder = Encoder(self.config)

    def forward(self, x: Tensor, output_hidden_states: Optional[bool] = None) -> BaseModelOutputWithNoAttention:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        embedding_output = self.embedder(x)
        encoder_outputs = self.encoder(embedding_output, output_hidden_states=output_hidden_states)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
        )

    def print_model_hash(self):
        print("Model parameters hashes:")
        for name, param in self.named_parameters():
            print(name, param.sum())
