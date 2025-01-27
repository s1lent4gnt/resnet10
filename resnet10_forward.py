from transformers import ResNetConfig, ResNetModel
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from torchvision.transforms import ToTensor


# class ResNetBlock(nn.Module):
#     """ResNet block."""

#     filters: int
#     conv: ModuleDef
#     norm: ModuleDef
#     act: Callable
#     strides: Tuple[int, int] = (1, 1)

#     @nn.compact
#     def __call__(
#         self,
#         x,
#     ):
#         residual = x
#         y = self.conv(self.filters, (3, 3), self.strides)(x)
#         y = self.norm()(y)
#         y = self.act(y)
#         y = self.conv(self.filters, (3, 3))(y)
#         y = self.norm()(y)

#         if residual.shape != y.shape:
#             residual = self.conv(self.filters, (1, 1), self.strides, name="conv_proj")(
#                 residual
#             )
#             residual = self.norm(name="norm_proj")(residual)

#         return self.act(residual + y)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, norm_groups=4):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels)
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
            nn.Conv2d(self.config.num_channels, self.config.embedding_size, kernel_size=7, stride=2, padding=3, bias=False),
            # class MyGroupNorm(nn.GroupNorm): - original code
            #     def __call__(self, x):
            #         if x.ndim == 3:
            #             x = x[jnp.newaxis]
            #             x = super().__call__(x)
            #             return x[0]
            #         else:
            #             return super().__call__(x)
            nn.GroupNorm(num_groups=4, eps=1e-5, num_channels=self.config.embedding_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.encoder = nn.Sequential(
            BasicBlock(self.config.embedding_size, 64),
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 512, stride=2)
        )

    def forward(self, x):
        out = self.embedder(x)
        out = self.encoder(out)

        return out