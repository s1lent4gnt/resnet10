import torch
import torch.nn as nn
from transformers import PreTrainedModel, ResNetConfig
from transformers.activations import ACT2FN
import numpy as np


def safe_convert_weights(jax_array, dtype=torch.float32):
    # First get the JAX array's dtype
    jax_dtype = jax_array.dtype
    
    # Convert to numpy while preserving dtype
    numpy_array = np.array(jax_array, dtype=jax_dtype)
    
    # Convert to torch tensor with specific dtype
    return torch.from_numpy(numpy_array).to(dtype)


class MyGroupNorm(nn.GroupNorm):
    """Custom GroupNorm that handles 3D inputs similar to the JAX implementation"""
    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
            x = super().forward(x)
            return x.squeeze(0)
        return super().forward(x)

class JaxStyleMaxPool(torch.nn.Module):
    def forward(self, x):
        x = torch.nn.functional.pad(x, (0, 1, 0, 1), value=-float('inf'))  # Pad right/bottom by 1
        return torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)

class BasicBlock(nn.Module):
    """ResNet basic block that matches the JAX implementation"""
    def __init__(self, in_channels, out_channels, activation, stride=1, norm_groups=4):
        super().__init__()
        
        # Match JAX initialization
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            dtype=torch.float32,
        )
        self.norm1 = MyGroupNorm(num_groups=norm_groups, num_channels=out_channels, eps=1e-5, affine=True, dtype=torch.float32)
        self.act1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False,
            dtype=torch.float32
        )
        self.norm2 = MyGroupNorm(num_groups=norm_groups, num_channels=out_channels, eps=1e-5, affine=True, dtype=torch.float32)
        self.act2 = nn.ReLU()

        # Only create shortcut if shapes don't match (like JAX implementation)
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, dtype=torch.float32),
                MyGroupNorm(num_groups=norm_groups, num_channels=out_channels, eps=1e-5, affine=True, dtype=torch.float32),
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        if self.shortcut is not None:
            print("TORCH before skip :", identity.mean().item())
            identity = self.shortcut[0](identity)
            print("TORCH after conv proj skip :", identity.mean().item())
            identity = self.shortcut[1](identity)
            print("TORCH after skip :", identity.mean().item())
            
        out = out + identity
        out = self.act2(out)
        
        return out


class ResNet10(PreTrainedModel):
    config_class = ResNetConfig
    
    def __init__(self, config):
        super().__init__(config)
        
        # Initial convolution and normalization (matches JAX implementation)
        self.embedder = nn.Sequential(
            nn.Conv2d(
                config.num_channels,
                config.embedding_size,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
                dtype=torch.float32,
            ),
            MyGroupNorm(num_groups=4, eps=1e-5, num_channels=config.embedding_size, affine=True, dtype=torch.float32),
            # ACT2FN[config.hidden_act],
            nn.ReLU(),
            JaxStyleMaxPool()
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # Encoder blocks
        self.encoder = nn.ModuleList()
        in_channels = config.embedding_size
        
        for i, size in enumerate(config.hidden_sizes):
            stride = 2 if i > 0 else 1  # First block doesn't downsample
            self.encoder.append(
                BasicBlock(
                    in_channels,
                    size,
                    activation=config.hidden_act,
                    stride=stride,
                )
            )
            in_channels = size


    def forward(self, x):
        print("TORCH Input: ", x.mean().item())
        x = self.embedder[0](x)
        print("TORCH After conv :", x.mean().item())
        x = self.embedder[1](x)
        print("TORCH After group norm :", x.mean().item())
        x = self.embedder[2](x)
        print("TORCH After activation :", x.mean().item())
        x = self.embedder[3](x)
        print("TORCH After max_pool :", x.mean().item())
        print("TORCH After embedder:", x.shape)
        for i, block in enumerate(self.encoder):
            x = block(x)
            print(f"TORCH After block {i}:", x.mean().item())
        
        return x

    def load_jax_weights(self, jax_state_dict):
        """Load only encoder weights (no output head)"""
        # Load initial conv layer
        self.embedder[0].load_state_dict({
            "weight": 
                safe_convert_weights(jax_state_dict["conv_init"]["kernel"]).permute(3, 2, 0, 1)  # JAX: (H, W, in_c, out_c) → PyTorch: (out_c, in_c, H, W)
        })
        
        # Load initial GroupNorm
        self.embedder[1].load_state_dict({
            "weight": safe_convert_weights(jax_state_dict["norm_init"]["scale"]),
            "bias": safe_convert_weights(jax_state_dict["norm_init"]["bias"])
        })

        # Load ResNet blocks
        for block_idx in range(4):  # For ResNetBlock_0 to ResNetBlock_3
            jax_block = jax_state_dict[f"ResNetBlock_{block_idx}"]
            torch_block = self.encoder[block_idx]
            
            # Conv layers
            torch_block.conv1.load_state_dict({
                "weight": 
                    safe_convert_weights(jax_block["Conv_0"]["kernel"]).permute(3, 2, 0, 1)
            })
            torch_block.conv2.load_state_dict({
                "weight": 
                    safe_convert_weights(jax_block["Conv_1"]["kernel"]).permute(3, 2, 0, 1)
            })

            # GroupNorm layers
            torch_block.norm1.load_state_dict({
                "weight": safe_convert_weights(jax_block["MyGroupNorm_0"]["scale"]),
                "bias": safe_convert_weights(jax_block["MyGroupNorm_0"]["bias"])
            })
            torch_block.norm2.load_state_dict({
                "weight": safe_convert_weights(jax_block["MyGroupNorm_1"]["scale"]),
                "bias": safe_convert_weights(jax_block["MyGroupNorm_1"]["bias"])
            })

            # Shortcut projections
            if "conv_proj" in jax_block:
                torch_block.shortcut[0].load_state_dict({
                    "weight": 
                        safe_convert_weights(jax_block["conv_proj"]["kernel"]).permute(3, 2, 0, 1)
                })
                torch_block.shortcut[1].load_state_dict({
                    "weight": safe_convert_weights(jax_block["norm_proj"]["scale"]),
                    "bias": safe_convert_weights(jax_block["norm_proj"]["bias"])
                })
