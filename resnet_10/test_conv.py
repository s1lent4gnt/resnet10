import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import jax.numpy as jnp
import flax
import jax

# class JaxStyleConv2d(nn.Module):
#     """Mimics JAX's Conv with padding='SAME' for exact parity."""
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
#         super().__init__()
#         self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
#         self.stride = stride if isinstance(stride, tuple) else (stride, stride)
#         self.conv = nn.Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size=self.kernel_size,
#             stride=self.stride,
#             padding=0,  # We handle padding manually
#             bias=bias,
#         )

#     def _compute_padding(self, input_height, input_width):
#         """Calculate asymmetric padding to match JAX's 'SAME' behavior."""
#         pad_h = max(0, (math.ceil(input_height / self.stride[0]) - 1) * self.stride[0] + self.kernel_size[0] - input_height)
#         pad_w = max(0, (math.ceil(input_width / self.stride[1]) - 1) * self.stride[1] + self.kernel_size[1] - input_width)

#         # Asymmetric padding (JAX adds extra padding to the right/bottom)
#         pad_top = pad_h // 2
#         pad_bottom = pad_h - pad_top
#         pad_left = pad_w // 2
#         pad_right = pad_w - pad_left

#         return (pad_left, pad_right, pad_top, pad_bottom)

#     def forward(self, x):
#         _, _, h, w = x.shape
#         pad_left, pad_right, pad_top, pad_bottom = self._compute_padding(h, w)
#         x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
#         return self.conv(x)


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


def get_jax_same_padding(h, w, k, s):
    """Compute asymmetric padding like JAX's SAME"""
    pad_h = max(0, (math.ceil(h / s) - 1) * s + k - h)
    pad_w = max(0, (math.ceil(w / s) - 1) * s + k - w)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return (pad_left, pad_right, pad_top, pad_bottom)

# Apply this to F.pad before passing to Conv2d
h, w = 128, 128  # Input height and width
kernel_size = 7
stride = 2
pad = get_jax_same_padding(h, w, kernel_size, stride)

x = torch.ones(1, 1, 128, 128)  # (N, C, H, W)
x_padded = F.pad(x, pad)  # Manually pad
conv = nn.Conv2d(1, 1, kernel_size=7, stride=2, padding=0, bias=False)  # Symmetric padding
ker = torch.ones_like(conv.weight)  # Make sure it has the same shape
conv.weight = torch.nn.Parameter(ker)
out_pt = conv(x_padded)  # Output shape: (1, 64, 3, 3)

# x = torch.randn(1, 3, 5, 5)  # (N, C, H, W)
conv_jax = JaxStyleConv2d(1, 1, kernel_size=7, stride=2)
ker_ = torch.ones_like(conv_jax.conv.weight)
conv_jax.conv.weight = torch.nn.Parameter(ker_)
out_jax_torch = conv_jax(x)  # Output shape: (1, 64, 3, 3)

x_jax = jnp.transpose(jnp.array(x.numpy(), dtype=jnp.float32), (0, 2, 3, 1))  # (N, H, W, C)
# 1. Initialize the module
# conv_module = flax.linen.Conv(features=1, kernel_size=(7, 7), strides=(2, 2), padding="SAME")
conv_module = flax.linen.Conv(features=1, kernel_size=(7, 7), strides=(2, 2), padding=[(3, 3), (3, 3)])

# 2. Initialize parameters with a random key
key = jax.random.PRNGKey(0)
variables = conv_module.init(key, x_jax)  # Returns {'params': ...}

variables["params"]["kernel"] = jnp.transpose(jnp.array(conv.weight.detach().numpy()), (3, 2, 0, 1))  # (N, H, W, C)

# 3. Apply the module with initialized parameters
out_jax = conv_module.apply(variables, x_jax)

# Input with known values (e.g., corners = 1.0, others = 0.0)
x = torch.zeros(1, 1, 5, 5)
x[0, 0, 0, 0] = 1.0  # Top-left corner
x[0, 0, -1, -1] = 1.0  # Bottom-right corner

# JAX-Style Conv
conv_jax = JaxStyleConv2d(1, 1, kernel_size=3, stride=2)
out_jax = conv_jax(x)

# PyTorch Default Conv
conv_pt = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
out_pt = conv_pt(x)