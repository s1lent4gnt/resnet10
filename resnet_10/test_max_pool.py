import jax.numpy as jnp
import flax.linen as nn  # JAX uses channel-last format
import torch
import numpy as np

# Create input tensor (5x5 grid)
input_data = np.random.normal(0, 100, size=(1, 32, 64, 64)) # NCHW
input_data = input_data.astype("float64")

# JAX setup (channel-last: NHWC)
# x_jax = input_data.reshape(1, 4, 4, 3).copy()  # (batch, height, width, channels)
x_jax = np.transpose(input_data, (0, 2, 3, 1)).copy()  # (batch, height, width, channels)
x_jax = x_jax.astype("float64")

# PyTorch setup (channel-first: NCHW)
x_torch = torch.tensor(input_data.copy(), dtype=torch.float64).reshape(1, 32, 64, 64)  # (batch, channels, height, width)

# JAX max pool (correct SAME padding implementation)
jax_pool = nn.max_pool(x_jax, (3, 3), strides=(2, 2), padding='SAME')

# With the manual asymmetric padding:
class JaxStyleMaxPool(torch.nn.Module):
    def forward(self, x):
        x = torch.nn.functional.pad(x, (0, 1, 0, 1), value=-float('inf'))  # Pad right/bottom by 1
        return torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)

torch_pool = JaxStyleMaxPool()
torch_result = torch_pool(x_torch)
# torch_result = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x_torch)
# torch_result = nn.max_pool(torch.tensor(input_data.copy(), dtype=torch.float16).reshape(1, 64, 64, 2).numpy(), (3, 3), strides=(2, 2), padding='SAME')
pass
