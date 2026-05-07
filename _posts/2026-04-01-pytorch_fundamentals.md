---
title: "Pytorch Fundamentals"
date: 2026-04-01
permalink: /posts/2026/04/pytorch_fundamentals/
tags:
  - pytorch
---

An introduction to PyTorch, covering tensor initialization, operations, indexing, and reshaping.


# PyTorch Fundamentals: Your First Steps into Hands-on Deep Learning

This notebook, derived from the [repository](https://github.com/analyticalrohit/pytorch_fundamentals), provides an introduction to PyTorch, covering tensor initialization, operations, indexing, and reshaping. Follow along to learn the basics with clear examples and detailed explanations.

# Table of Contents

- [What are Tensors?](#What-are-Tensors?)
- [Tensor Initialization](#Tensor-Initialization)
- [Common Tensor Initialization Methods](#Common-Tensor-Initialization-Methods)
- [Tensor Type Conversion](#Tensor-Type-Conversion)
- [Converting Between NumPy Arrays and Tensors](#Converting-Between-NumPy-Arrays-and-Tensors)
- [Tensor Mathematics and Comparison Operations](#Tensor-Mathematics-and-Comparison-Operations)
- [Matrix Multiplication and Batch Operations](#Matrix-Multiplication-and-Batch-Operations)
- [Broadcasting and Other Useful Operations](#Broadcasting-and-Other-Useful-Operations)
- [Tensor Indexing](#Tensor-Indexing)
- [Tensor Reshaping](#Tensor-Reshaping)


```python
import torch
import numpy as np
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Print versions
print("torch version:", torch.__version__)
print("numpy version:", np.__version__)
```

    torch version: 2.4.0+cu118
    numpy version: 1.26.3


## What are Tensors?

Tensor holds a multi-dimensional array of elements of a single data type which is very similar with numpy’s ndarray. When the dimension is zero, it can be called a scalar. When the dimension is 1, it can be called a vector. When the dimension is 2, it can be called a matrix. 

- 0-dimensional tensor: A single number (scalar).
- 1-dimensional tensor: A list of numbers (vector).
- 2-dimensional tensor: A table of numbers (matrix).

When the dimension is greater than 2, it is usually called a tensor. 

<img src="data:image/webp;base64,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" alt="Logo" style="width:800px;">

对tensor的理解：

> 解释：torch.Size([B,T,H,W,D])
> 
> 这是一个五维张量，包含了B个四维张量，
> 
> 每个四维张量包含了T个三维张量，
> 
> 每个三维张量包含了H个二维张量，
> 
> 每个二维张量包含了W个一维张量，
> 
> 每个一维张量包含了D个元素。

## Tensor Initialization

This code creates a 2×3 PyTorch tensor with float32 data type, assigns it to a specified device (CPU or GPU), and enables gradient tracking for backpropagation. 


```python
# Check for CUDA availability and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize a 2x3 tensor with requires_grad enabled
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True)

print(my_tensor)
print("Data type:", my_tensor.dtype)
print("Device:", my_tensor.device)
print("Dim:", my_tensor.ndim)
print("Shape:", my_tensor.shape)
print("Requires Gradient:", my_tensor.requires_grad)
```

    tensor([[1., 2., 3.],
            [4., 5., 6.]], device='cuda:0', requires_grad=True)
    Data type: torch.float32
    Device: cuda:0
    Dim: 2
    Shape: torch.Size([2, 3])
    Requires Gradient: True


## Other Common Tensor Initialization Methods

- **Empty Tensor:** Creates an uninitialized 3×3 tensor (random values).
- **Zeros Tensor:** Creates a 3×3 tensor filled with zeros.
- **Random Tensor:** Generates a 3×3 tensor with random values between 0 and 1.
- **Ones Tensor:** Creates a 3×3 tensor filled with ones.
- **Identity Matrix:** Generates a 4×4 identity matrix (diagonal of ones).
- **Arange Tensor:** Creates a 1D tensor with values from 0 to 4 (step of 1).
- **Linspace Tensor:** Generates 5 evenly spaced values between 0.1 and 1.
- **Normal Distributed Tensor:** Fills a tensor with values from a normal (Gaussian) distribution with mean 0 and std 1.
- **Uniform Distributed Tensor:** Fills a tensor with values from a uniform distribution between 0 and 1.
- **Diagonal Tensor:** Creates a 4×4 diagonal tensor with ones along the diagonal and zeros elsewhere.


```python
# Create an empty tensor of size 3x3
x = torch.empty(3, 3)
print("Empty Tensor:\n", x)

# Create a tensor filled with zeros
x = torch.zeros(3, 3)
print("Zeros Tensor:\n", x)

# Create a tensor with random values
x = torch.rand(3, 3)
print("Random Tensor:\n", x)

# Create a tensor filled with ones
x = torch.ones(3, 3)
print("Ones Tensor:\n", x)

# Create an identity matrix
x = torch.eye(4, 4)
print("Identity Matrix:\n", x)

# Create a tensor using arange
x = torch.arange(5)
print("Arange Tensor:\n", x)

# Create a tensor using linspace
x = torch.linspace(0.1, 1, 5)
print("Linspace Tensor:\n", x)

# Create a tensor with values drawn from a normal distribution
x = torch.empty(1, 5).normal_(mean=0, std=1)
print("Normal Distributed Tensor:\n", x)

# Create a tensor with values drawn from a uniform distribution
x = torch.empty(1, 5).uniform_(0, 1)
print("Uniform Distributed Tensor:\n", x)

# Create a diagonal tensor from a tensor of ones
x = torch.diag(torch.ones(4))
print("Diagonal Tensor:\n", x)
```

    Empty Tensor:
     tensor([[0.2450, 0.9899, 0.0546],
            [0.4938, 0.7471, 0.5465],
            [0.0106, 0.3488, 0.2002]])
    Zeros Tensor:
     tensor([[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]])
    Random Tensor:
     tensor([[0.1397, 0.6464, 0.0529],
            [0.5808, 0.0736, 0.0454],
            [0.7504, 0.4951, 0.9513]])
    Ones Tensor:
     tensor([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]])
    Identity Matrix:
     tensor([[1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]])
    Arange Tensor:
     tensor([0, 1, 2, 3, 4])
    Linspace Tensor:
     tensor([0.1000, 0.3250, 0.5500, 0.7750, 1.0000])
    Normal Distributed Tensor:
     tensor([[-1.2343, -0.4473,  0.6663,  0.0808,  0.3924]])
    Uniform Distributed Tensor:
     tensor([[0.4357, 0.9985, 0.8195, 0.4854, 0.3971]])
    Diagonal Tensor:
     tensor([[1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]])


## Tensor Type Conversion

Creates a tensor with values [0, 1, 2, 3] and demonstrates type conversion to boolean, int16, int64, float16, float32, and float64.


```python
# Create a tensor and convert its type
tensor = torch.arange(4)
print("Boolean Tensor:", tensor.bool())   # Convert to boolean
print("Short Tensor (int16):", tensor.short())   # Convert to int16
print("Long Tensor (int64):", tensor.long())   # Convert to int64
print("Half Tensor (float16):", tensor.half())   # Convert to float16
print("Float Tensor (float32):", tensor.float())   # Convert to float32
print("Double Tensor (float64):", tensor.double())   # Convert to float64
```

    Boolean Tensor: tensor([False,  True,  True,  True])
    Short Tensor (int16): tensor([0, 1, 2, 3], dtype=torch.int16)
    Long Tensor (int64): tensor([0, 1, 2, 3])
    Half Tensor (float16): tensor([0., 1., 2., 3.], dtype=torch.float16)
    Float Tensor (float32): tensor([0., 1., 2., 3.])
    Double Tensor (float64): tensor([0., 1., 2., 3.], dtype=torch.float64)


## Converting Between NumPy Arrays and Tensors

PyTorch makes it easy to switch between NumPy arrays and tensors, allowing seamless integration with existing computing workflows.


```python
# Create a NumPy array of zeros
np_array = np.zeros((5, 5))
print("NumPy Array:\n", np_array)

# Convert NumPy array to PyTorch tensor
tensor = torch.from_numpy(np_array)
print("Tensor from NumPy Array:\n", tensor)

# Convert tensor back to NumPy array
numpy_back = tensor.numpy()
print("Converted Back to NumPy Array:\n", numpy_back)
```

    NumPy Array:
     [[0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]]
    Tensor from NumPy Array:
     tensor([[0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]], dtype=torch.float64)
    Converted Back to NumPy Array:
     [[0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]]


## Tensor Mathematics and Comparison Operations

This section explores essential math operations with PyTorch tensors. 

- **Addition & Subtraction:** Adds and subtracts two tensors element-wise.  
- **Division:** Uses true division for precise results.  
- **Inplace Operations:** Modifies a tensor directly without creating a new one.  
- **Exponentiation:** Raises each element to a power using `pow()` or `**`.  
- **Comparisons:** Checks conditions like `x > 0` or `x < 0`, returning boolean results.  
- **Dot Product:** Computes the sum of element-wise multiplications between two tensors. 


```python
# Define two tensors for operations
x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# Addition
z = x + y
print("Addition Results:", z)

# Addition using .add
z1 = torch.empty(3)
torch.add(x, y, out=z1)
z2 = torch.add(x, y)
print("Addition Results:", z, z1, z2)

# Subtraction
z = x - y
print("Subtraction Result:", z)

# Division (true division)
z = torch.true_divide(x, y)
print("Division Result:", z)

# Inplace operations
t = torch.ones(3)
print("Before inplace addition:", t)
t.add_(x)
print("After inplace addition:", t)
t += x  # Another inplace addition (note: t = t + x creates a new tensor)
print("After second inplace addition:", t)

# Exponentiation
z = x.pow(2)
print("Exponentiation (pow):", z)
z = x**2
print("Exponentiation (**):", z)

# Comparisons
z = x > 0
print("x > 0:", z)
z = x < 0
print("x < 0:", z)

# Dot product
z = torch.dot(x, y)
print("Dot Product:", z)
```

    Addition Results: tensor([10, 10, 10])
    Addition Results: tensor([10, 10, 10]) tensor([10., 10., 10.]) tensor([10, 10, 10])
    Subtraction Result: tensor([-8, -6, -4])
    Division Result: tensor([0.1111, 0.2500, 0.4286])
    Before inplace addition: tensor([1., 1., 1.])
    After inplace addition: tensor([2., 3., 4.])
    After second inplace addition: tensor([3., 5., 7.])
    Exponentiation (pow): tensor([1, 4, 9])
    Exponentiation (**): tensor([1, 4, 9])
    x > 0: tensor([True, True, True])
    x < 0: tensor([False, False, False])
    Dot Product: tensor(46)


## Matrix Multiplication and Batch Operations

Matrix operations are at the heart of deep learning. Let's find out different ways to perform multiplication.

- **Matrix Multiplication:** Uses `@` or `torch.mm()` to perform standard matrix multiplication.  
- **Matrix Exponentiation:** Raises a square matrix to a power using `matrix_power(n)`.  
- **Element-wise Multiplication:** Uses `torch.mul()` or `*` for element-wise multiplication.  
- **Batch Matrix Multiplication:** Uses `torch.bmm()` to multiply batches of matrices efficiently.


```python
# Matrix multiplication using @ operator and torch.mm
x2 = torch.tensor([[1, 2, 3]])
y2 = torch.tensor([[9, 8, 7]])

z = x2 @ torch.t(y2)
print("Matrix Multiplication (@ operator):\n", z)
z = torch.mm(x2, torch.t(y2))
print("Matrix Multiplication (torch.mm):\n", z)
z = x2.mm(torch.t(y2))
print("Matrix Multiplication (mm):\n", z)

# Matrix exponentiation: multiplying a matrix with itself 3 times
matrix_exp = torch.rand(5, 5)
print("Matrix multiplied 3 times:\n", matrix_exp @ matrix_exp @ matrix_exp)
print("Matrix power 3:\n", matrix_exp.matrix_power(3))

# Element-wise multiplication
z = torch.mul(x, y)
print("Element-wise Multiplication:", z)
z = x * y
print("Element-wise Multiplication (alternative):", z)

# Batch matrix multiplication
batch = 32
n, m, p = 10, 20, 30
tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2)  # Result shape: (batch, n, p)
print("Batch Matrix Multiplication (first batch):\n", out_bmm[0])
print("Shape of batched multiplication result:", (tensor1 @ tensor2).shape)
```

    Matrix Multiplication (@ operator):
     tensor([[46]])
    Matrix Multiplication (torch.mm):
     tensor([[46]])
    Matrix Multiplication (mm):
     tensor([[46]])
    Matrix multiplied 3 times:
     tensor([[6.8649, 7.1375, 6.5783, 7.6163, 6.8593],
            [6.4644, 6.5831, 6.2444, 7.0148, 6.4003],
            [5.8586, 6.0872, 5.5466, 6.2068, 5.5633],
            [6.6579, 6.8326, 6.3734, 7.4843, 6.8450],
            [5.6023, 5.7104, 5.4097, 5.7048, 5.1312]])
    Matrix power 3:
     tensor([[6.8649, 7.1375, 6.5783, 7.6163, 6.8593],
            [6.4644, 6.5831, 6.2444, 7.0148, 6.4003],
            [5.8586, 6.0872, 5.5466, 6.2068, 5.5633],
            [6.6579, 6.8326, 6.3734, 7.4843, 6.8450],
            [5.6023, 5.7104, 5.4097, 5.7048, 5.1312]])
    Element-wise Multiplication: tensor([ 9, 16, 21])
    Element-wise Multiplication (alternative): tensor([ 9, 16, 21])
    Batch Matrix Multiplication (first batch):
     tensor([[3.5440, 6.5026, 6.5445, 4.5252, 5.2045, 5.7321, 6.1181, 6.6445, 6.0093,
             4.9353, 3.9677, 4.6019, 6.3319, 5.7043, 5.1247, 4.3091, 4.7326, 4.8858,
             5.1322, 5.6088, 5.9398, 6.9429, 5.9886, 5.2573, 2.7003, 4.8218, 5.9894,
             4.6512, 5.1542, 3.7731],
            [4.2282, 5.0751, 4.5581, 3.9363, 5.2671, 5.2506, 4.4924, 5.5095, 5.1781,
             3.7145, 4.7497, 3.8810, 4.2245, 4.6506, 4.5174, 2.8565, 3.9761, 3.8779,
             4.3638, 4.0318, 5.5015, 3.9265, 5.3370, 4.7982, 2.2797, 4.3291, 3.7386,
             4.0959, 3.6940, 3.2689],
            [4.5827, 6.1946, 4.9569, 4.6735, 5.6490, 5.8873, 5.7623, 6.4767, 6.3909,
             4.4470, 5.0609, 4.6619, 5.1481, 5.3259, 5.4949, 3.7388, 5.1354, 4.2429,
             4.5741, 4.5234, 5.5854, 4.6225, 5.9995, 5.5425, 2.8380, 4.2748, 4.3472,
             4.4736, 4.1825, 3.4843],
            [3.3100, 5.1634, 5.0415, 3.7873, 4.4239, 5.4084, 5.0781, 5.1528, 5.1801,
             3.5463, 3.6643, 3.9544, 4.7713, 4.8103, 3.8891, 2.8634, 4.0375, 3.4439,
             4.1778, 4.0538, 4.9986, 4.8988, 5.2553, 4.8321, 1.6977, 4.3844, 4.1179,
             4.2077, 4.1596, 3.0347],
            [4.1352, 5.5097, 5.4081, 4.1339, 5.2045, 5.8074, 5.9689, 5.7145, 5.1557,
             4.5157, 4.7473, 5.3542, 5.0027, 4.4979, 5.2767, 4.4354, 3.5934, 4.7134,
             5.1901, 5.5433, 6.2373, 4.5942, 5.5256, 5.3336, 3.2117, 4.1416, 4.2712,
             4.1048, 4.0465, 3.6511],
            [3.7322, 5.3081, 5.4181, 4.6393, 5.1372, 5.5894, 5.5543, 6.0296, 4.5670,
             4.0552, 4.2119, 4.5391, 5.2059, 4.8795, 5.0137, 3.7978, 3.7304, 4.0530,
             4.3719, 5.0318, 5.7526, 4.8653, 4.8926, 5.5795, 2.6404, 4.2889, 4.5551,
             4.2022, 3.9910, 3.8555],
            [3.2315, 4.6632, 4.9304, 3.5092, 4.4533, 4.8799, 4.6290, 5.3459, 4.9640,
             3.7996, 4.0911, 4.0296, 4.6814, 4.0135, 4.5876, 3.6069, 3.8043, 4.2701,
             4.1166, 4.8829, 5.5559, 4.8536, 4.5921, 3.6704, 2.4689, 3.4953, 3.6832,
             3.9577, 3.5755, 3.0293],
            [4.7066, 5.7029, 6.1439, 4.4224, 5.7423, 6.4657, 5.8944, 5.8826, 6.2143,
             4.1787, 4.8364, 4.9471, 5.2735, 4.8664, 5.5245, 3.1057, 4.8749, 4.4022,
             5.6390, 5.2407, 6.2896, 5.6143, 5.5157, 5.6265, 2.1874, 4.4567, 3.9869,
             5.5333, 4.4775, 3.9314],
            [4.3227, 5.8580, 5.9357, 4.1483, 5.2436, 6.3560, 5.7144, 5.9563, 5.8249,
             4.5020, 4.4281, 4.4165, 5.4092, 4.8056, 4.9667, 4.0844, 4.2222, 3.8133,
             4.5904, 5.0534, 6.7042, 5.0727, 5.5185, 5.6069, 3.0032, 4.5262, 4.0067,
             4.7430, 4.2932, 3.7362],
            [3.8294, 5.4057, 5.0970, 3.9448, 5.0699, 5.1357, 4.9649, 5.2615, 4.4683,
             3.5833, 4.0856, 3.3753, 4.0069, 4.1184, 4.2750, 3.4178, 3.7880, 3.3670,
             4.7347, 4.6245, 5.3556, 3.9735, 5.1346, 5.1036, 2.6921, 4.2103, 3.9386,
             3.6596, 4.1051, 2.2051]])
    Shape of batched multiplication result: torch.Size([32, 10, 30])

‌高维矩阵（张量Tensor）的矩阵乘法
**核心原理‌：** 将最后两个维度视为标准二维矩阵，其余维度作为“批量维度”（batch dimensions），执行批量矩阵乘法。
**形状要求‌：**
设两个张量形状分别为 (..., M, K) 和 (..., K, N)，则结果形状为 (..., M, N)。
**广播规则‌：** 
广播仅作用于最后两个维度之前的维度‌（即批量维度）。
（从后向前对齐）：
对应维度相等，或
其中一个维度为 1，或
缺失维度视为 1。
若不满足上述条件，则抛出广播错误。

## Broadcasting and Other Useful Operations

Broadcasting allows arithmetic operations on tensors of different shapes. This section also demonstrates additional useful functions.

- **Broadcasting:** Automatically expands smaller tensors to match larger ones in operations.  
- **Summation:** `torch.sum(x, dim=0)` computes sum along a specific dimension.  
- **Min/Max Values:** `torch.max()` and `torch.min()` return the highest and lowest values along a dimension.  
- **Absolute Values:** `torch.abs(x)` gets the element-wise absolute values.  
- **Argmax/Argmin:** `torch.argmax()` and `torch.argmin()` return the index of max/min values.  
- **Mean Calculation:** `torch.mean(x.float(), dim=0)` computes the mean (ensuring float dtype).  
- **Element-wise Comparison:** `torch.eq(x, y)` checks equality between two tensors.  
- **Sorting:** `torch.sort(y, dim=0)` sorts tensor elements and returns indices.  
- **Clamping:** `torch.clamp(x, min=0)` restricts values within a range.  
- **Boolean Operations:** `torch.any(x_bool)` checks if any value is `True`, `torch.all(x_bool)` checks if all are `True`.  


```python
# Broadcasting example
x1 = torch.rand(5, 5)
x2 = torch.rand(5)
print("Tensor x1:\n", x1)
print("Tensor x2:\n", x2)
print("x1 - x2:\n", x1 - x2)
print("x1 raised to the power of x2:\n", x1 ** x2)

# Sum of tensor elements along dimension 0
sum_x = torch.sum(x, dim=0)
print("Sum along dimension 0:", sum_x)

# Maximum and minimum values
value, indices = torch.max(x, dim=0)
print("Max value and index:", value, indices)

value, indices = torch.min(x, dim=0)
print("Min value and index:", value, indices)

# Other operations
print("Absolute values:", torch.abs(x))
print("Argmax:", torch.argmax(x, dim=0))
print("Argmin:", torch.argmin(x, dim=0))
print("Mean (converted to float):", torch.mean(x.float(), dim=0))
print("Element-wise equality (x == y):", torch.eq(x, y))

# Sorting
sorted_y, indices = torch.sort(y, dim=0, descending=False)
print("Sorted y and indices:", sorted_y, indices)

# Clamping values
print("Clamped x:", torch.clamp(x, min=0))

# Boolean operations
x_bool = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)
print("Any True:", torch.any(x_bool))
print("All True:", torch.all(x_bool))
```

    Tensor x1:
     tensor([[0.5610, 0.6928, 0.8066, 0.2603, 0.1528],
            [0.9867, 0.1102, 0.4665, 0.1929, 0.6669],
            [0.4671, 0.0768, 0.6585, 0.5024, 0.8904],
            [0.6634, 0.6646, 0.0860, 0.1698, 0.0833],
            [0.5990, 0.0964, 0.4688, 0.3539, 0.0450]])
    Tensor x2:
     tensor([0.8125, 0.2102, 0.6983, 0.5526, 0.1509])
    x1 - x2:
     tensor([[-0.2516,  0.4826,  0.1083, -0.2923,  0.0019],
            [ 0.1741, -0.1000, -0.2318, -0.3597,  0.5160],
            [-0.3454, -0.1334, -0.0397, -0.0503,  0.7396],
            [-0.1491,  0.4544, -0.6123, -0.3828, -0.0676],
            [-0.2135, -0.1138, -0.2294, -0.1988, -0.1058]])
    x1 raised to the power of x2:
     tensor([[0.6252, 0.9258, 0.8606, 0.4753, 0.7532],
            [0.9892, 0.6290, 0.5872, 0.4028, 0.9407],
            [0.5388, 0.5831, 0.7470, 0.6836, 0.9826],
            [0.7164, 0.9177, 0.1803, 0.3754, 0.6873],
            [0.6594, 0.6116, 0.5892, 0.5632, 0.6264]])
    Sum along dimension 0: tensor(6)
    Max value and index: tensor(3) tensor(2)
    Min value and index: tensor(1) tensor(0)
    Absolute values: tensor([1, 2, 3])
    Argmax: tensor(2)
    Argmin: tensor(0)
    Mean (converted to float): tensor(2.)
    Element-wise equality (x == y): tensor([False, False, False])
    Sorted y and indices: tensor([7, 8, 9]) tensor([2, 1, 0])
    Clamped x: tensor([1, 2, 3])
    Any True: tensor(True)
    All True: tensor(False)


## Tensor Indexing

Access and modify tensor elements using indexing, slicing, and advanced indexing.

- **Accessing Rows & Columns:** Use `x[row, :]` for a row and `x[:, col]` for a column.  
- **Slicing:** `x[row, start:end]` extracts a portion of a row.  
- **Modifying Elements:** Directly assign values using `x[row, col] = value`.  
- **Fancy Indexing:** Use a list of indices to select multiple elements at once.  
- **Conditional Indexing:** Extract elements using conditions like `(x < 2) | (x > 8)`.  
- **Finding Even Numbers:** Use `x.remainder(2) == 0` to filter even values.  
- **Conditional Selection with `torch.where()`:** Chooses values based on a condition.  


```python
# Create a random tensor with shape (batch_size, features)
batch_size = 10
features = 25
x = torch.rand((batch_size, features))

# Access the first row
print("First row of tensor:", x[0, :])

# Access the second column
print("Second column of tensor:", x[:, 1])

# Access the first 10 elements of the third row
print("First 10 elements of third row:", x[2, 0:10])

# Modify a specific element (set first element to 100)
x[0, 0] = 100

# Fancy indexing example
x1 = torch.arange(10)
indices = [2, 5, 8]
print("Fancy indexing result:", x1[indices])

# Advanced indexing: select elements based on a condition
x2 = torch.arange(10)
print("Elements where x2 < 2 or x2 > 8:", x2[(x2 < 2) | (x2 > 8)])
print("Even numbers in x2:", x2[x2.remainder(2) == 0])

# Using torch.where to select values based on a condition
'''
torch.where(condition, x, y)
- condition：布尔型张量，决定从 x 或 y 中选择元素
- x：condition 为 True 时选择的值，支持张量或标量
- y：condition 为 False 时选择的值，支持张量或标量
'''
print("Using torch.where:", torch.where(x2 > 5, x2, x2 * 2))


```

    First row of tensor: tensor([0.1933, 0.0269, 0.3945, 0.6182, 0.3705, 0.7060, 0.4922, 0.0280, 0.3398,
            0.9600, 0.2417, 0.8861, 0.1833, 0.0985, 0.2710, 0.6410, 0.3799, 0.5981,
            0.0205, 0.9136, 0.9481, 0.6899, 0.9450, 0.6970, 0.1787])
    Second column of tensor: tensor([2.6887e-02, 8.1319e-02, 9.8993e-01, 4.5033e-01, 5.7220e-04, 9.5527e-01,
            1.1555e-01, 9.4050e-02, 5.3863e-02, 5.0582e-01])
    First 10 elements of third row: tensor([0.2450, 0.9899, 0.0546, 0.4938, 0.7471, 0.5465, 0.0106, 0.3488, 0.2002,
            0.4488])
    Fancy indexing result: tensor([2, 5, 8])
    Elements where x2 < 2 or x2 > 8: tensor([0, 1, 9])
    Even numbers in x2: tensor([0, 2, 4, 6, 8])
    Using torch.where: tensor([ 0,  2,  4,  6,  8, 10,  6,  7,  8,  9])


## Tensor Reshaping

Learn how to reshape tensors, concatenate them, and change the order of dimensions.

- **Reshape with `view()` & `reshape()`:** Change tensor shape without altering data.  
- **Transpose & Flatten:** `.t()` transposes, `.contiguous().view(-1)` flattens.  
- **Concatenation:** `torch.cat([x1, x2], dim=0/1)` merges tensors along rows/columns.  
- **Flattening:** `.view(-1)` converts a tensor into a 1D array.  
- **Batch Reshaping:** `.view(batch, -1)` keeps batch size while reshaping.  
- **Permute Dimensions:** `.permute(0, 2, 1)` reorders dimensions efficiently.  
- **Unsqueeze for New Dimensions:** `.unsqueeze(dim)` adds singleton dimensions.  


```python
# Reshape a tensor using view and reshape
x = torch.arange(9)
x_3x3 = x.view(3, 3)
print("Reshaped to 3x3 using view:\n", x_3x3)
x_3x3 = x.reshape(3, 3)
print("Reshaped to 3x3 using reshape:\n", x_3x3)

# Transpose and flatten the tensor
y = x_3x3.t()
print("Flattened transposed tensor:", y.contiguous().view(9))

# Concatenation example
x1 = torch.rand(2, 5)
x2 = torch.rand(2, 5)
print("Concatenated along dimension 0 (rows):", torch.cat([x1, x2], dim=0).shape)
print("Concatenated along dimension 1 (columns):", torch.cat([x1, x2], dim=1).shape)

# Flatten the tensor using view(-1)
z = x1.view(-1)
print("Flattened tensor shape:", z.shape)

# Reshape with batch dimension
batch = 64
x = torch.rand(batch, 2, 5)
print("Reshaped to (batch, -1):", x.view(batch, -1).shape)

# Permute dimensions
z = x.permute(0, 2, 1)
print("Permuted tensor shape:", z.shape)

# Unsqueeze examples (adding new dimensions)
x = torch.arange(10)
print("Original x:", x)
print("x unsqueezed at dim 0:", x.unsqueeze(0).shape, x.unsqueeze(0))
print("x unsqueezed at dim 1:", x.unsqueeze(1).shape, x.unsqueeze(1))

```

    Reshaped to 3x3 using view:
     tensor([[0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]])
    Reshaped to 3x3 using reshape:
     tensor([[0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]])
    Flattened transposed tensor: tensor([0, 3, 6, 1, 4, 7, 2, 5, 8])
    Concatenated along dimension 0 (rows): torch.Size([4, 5])
    Concatenated along dimension 1 (columns): torch.Size([2, 10])
    Flattened tensor shape: torch.Size([10])
    Reshaped to (batch, -1): torch.Size([64, 10])
    Permuted tensor shape: torch.Size([64, 5, 2])
    Original x: tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    x unsqueezed at dim 0: torch.Size([1, 10]) tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    x unsqueezed at dim 1: torch.Size([10, 1]) tensor([[0],
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [7],
            [8],
            [9]])

