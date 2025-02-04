import torch
import torch.nn as nn
from transformers import PreTrainedModel, ResNetConfig
from transformers.activations import ACT2FN
import numpy as np
import math
import torch.nn.functional as F


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
    
class JaxStyleConv2d(nn.Module):
    """Mimics JAX's Conv2D with padding='SAME' for exact parity."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False):
        super().__init__()
        
        # Ensure kernel_size and stride are tuples
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        
        # Create convolution layer with padding=0 (manual padding will be applied)
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            padding=0,  # No built-in padding
            bias=bias
        )

    def _compute_padding(self, input_height, input_width):
        """Calculate asymmetric padding to match JAX's 'SAME' behavior."""
        
        # Compute padding needed for height and width
        pad_h = max(0, (math.ceil(input_height / self.stride[0]) - 1) * self.stride[0] + self.kernel_size[0] - input_height)
        pad_w = max(0, (math.ceil(input_width / self.stride[1]) - 1) * self.stride[1] + self.kernel_size[1] - input_width)

        # Asymmetric padding (JAX-style: more padding on the bottom/right if needed)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        return (pad_left, pad_right, pad_top, pad_bottom)

    def forward(self, x):
        """Apply asymmetric padding before convolution."""
        _, _, h, w = x.shape  # Get input height and width
        
        # Compute asymmetric padding
        pad_left, pad_right, pad_top, pad_bottom = self._compute_padding(h, w)
        
        # Apply manual padding (PyTorch format: (left, right, top, bottom))
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        
        # Apply convolution
        return self.conv(x)

class BasicBlock(nn.Module):
    """ResNet basic block that matches the JAX implementation"""
    def __init__(self, in_channels, out_channels, activation, stride=1, norm_groups=4):
        super().__init__()
        
        # Match JAX initialization
        self.conv1 = JaxStyleConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            bias=False,
        )
        self.norm1 = MyGroupNorm(num_groups=norm_groups, num_channels=out_channels, eps=1e-5, affine=True)
        self.act1 = nn.ReLU()
        
        self.conv2 = JaxStyleConv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1,  
            bias=False,
        )
        self.norm2 = MyGroupNorm(num_groups=norm_groups, num_channels=out_channels, eps=1e-5, affine=True)
        self.act2 = nn.ReLU()

        # Only create shortcut if shapes don't match (like JAX implementation)
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                JaxStyleConv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                MyGroupNorm(num_groups=norm_groups, num_channels=out_channels, eps=1e-5, affine=True),
            )

    def forward(self, x):
        identity = x
        
        print("TORCH input to block :", x.mean().item())
        out = self.conv1(x)
        print("TORCH conv1 :", out.mean().item())
        out = self.norm1(out)
        print("TORCH norm1:", out.mean().item())
        out = self.act1(out)
        print("TORCH act1 :", out.mean().item())
        
        out = self.conv2(out)
        print("TORCH conv2 :", out.mean().item())
        out = self.norm2(out)
        print("TORCH norm2 :", out.mean().item())
        
        if self.shortcut is not None:
            print("TORCH before skip :", identity.mean().item())
            identity = self.shortcut[0](identity)
            print("TORCH after conv proj skip :", identity.mean().item())
            identity = self.shortcut[1](identity)
            print("TORCH after skip :", identity.mean().item())

        print("TORCH before **** sum :", out.mean().item())
        out = out + identity
        print("TORCH after sum :", out.mean().item())
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
                "conv.weight": 
                    safe_convert_weights(jax_block["Conv_0"]["kernel"]).permute(3, 2, 0, 1)
            })
            torch_block.conv2.load_state_dict({
                "conv.weight": 
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
                    "conv.weight": 
                        safe_convert_weights(jax_block["conv_proj"]["kernel"]).permute(3, 2, 0, 1)
                })
                torch_block.shortcut[1].load_state_dict({
                    "weight": safe_convert_weights(jax_block["norm_proj"]["scale"]),
                    "bias": safe_convert_weights(jax_block["norm_proj"]["bias"])
                })
