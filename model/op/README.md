# Pure Python Operations

This directory contains pure Python implementations of operations that were previously implemented using CUDA and C++ extensions.

## Operations

### FusedLeakyReLU

A fused operation that combines bias addition and leaky ReLU activation. The implementation is in `fused_act_py.py`.

```python
from model.op import FusedLeakyReLU, fused_leaky_relu

# As a module
layer = FusedLeakyReLU(channels=64)
output = layer(input)

# As a function
output = fused_leaky_relu(input, bias, negative_slope=0.2, scale=2**0.5)
```

### UpFirDn2D

An operation for upsampling, FIR filtering, and downsampling 2D data. The implementation is in `upfirdn2d_py.py`.

```python
from model.op import upfirdn2d
import torch

# Create a kernel (e.g., a Gaussian blur kernel)
kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) / 16.0

# Apply upfirdn2d
# up=2: upsample by a factor of 2
# down=1: no downsampling
# pad=(1, 1): padding of 1 on all sides
output = upfirdn2d(input, kernel, up=2, down=1, pad=(1, 1))
```

## Advantages of Pure Python Implementation

1. **No CUDA/C++ dependencies**: The code works without requiring CUDA or C++ compilation.
2. **Cross-platform compatibility**: Works on any platform that supports PyTorch.
3. **Easier to understand and modify**: Pure Python code is more accessible for understanding and modification.
4. **Automatic differentiation**: PyTorch's autograd handles the gradients automatically.

## Performance Considerations

The pure Python implementations may be slower than the CUDA versions, especially for large inputs. If performance is critical, you might want to consider using the original CUDA implementations when available. 