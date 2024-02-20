# NumPy Library Overview

NumPy is a Python library used for numerical computing, particularly for handling large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently. It's an essential tool for scientific computing, data analysis, and machine learning tasks.

## Benefits of NumPy:

1. **Efficiency:** NumPy operations are optimized and implemented in C, making them much faster compared to traditional Python lists, especially for large datasets.
2. **Multi-dimensional array support:** NumPy provides powerful tools for working with arrays of any dimension, facilitating complex mathematical operations and manipulations.
3. **Broadcasting:** NumPy enables operations between arrays of different shapes and sizes through broadcasting, which simplifies code and improves readability.
4. **Comprehensive mathematical functions:** It offers a wide range of mathematical functions for array manipulation, linear algebra, Fourier transforms, statistics, and more, making it versatile for various scientific computing tasks.
5. **Integration with other libraries:** NumPy seamlessly integrates with other Python libraries like SciPy (for scientific computing), Matplotlib (for data visualization), and pandas (for data manipulation), forming a powerful ecosystem for data analysis and computation.

## Drawbacks of NumPy:

1. **Steep learning curve:** Understanding NumPy's array manipulation techniques and broadcasting rules may require some time and effort, especially for beginners.
2. **Memory consumption:** NumPy arrays can consume more memory than traditional Python lists due to their fixed data type and the need for contiguous memory allocation.

## Comparison between NumPy Array and Normal List:

| Feature               | NumPy Array                                   | Normal List                                |
|-----------------------|-----------------------------------------------|--------------------------------------------|
| Memory Efficiency     | More memory-efficient due to homogeneous data types and contiguous memory blocks. | Less memory-efficient, as elements can have different data types and are scattered in memory. |
| Performance           | Faster for numerical computations due to optimized C implementations. | Slower for numerical computations, especially with large datasets. |
| Functionality         | Provides specialized functions for numerical computing and array manipulation. | Offers general-purpose data storage and manipulation capabilities. |
| Convenience           | Simplifies mathematical operations and manipulations on arrays. | May require more verbose code for similar operations. |

## Usage in AI:

In the field of Artificial Intelligence (AI), NumPy is indispensable for various tasks such as:
- **Data preprocessing:** NumPy is used to manipulate and prepare datasets for training machine learning models.
- **Linear algebra:** Many AI algorithms, including neural networks, rely heavily on linear algebra operations, which are efficiently handled by NumPy.
- **Numerical computations:** NumPy's speed and efficiency make it ideal for performing complex numerical computations required in AI research and development.
- **Model evaluation:** NumPy provides tools for evaluating the performance of machine learning models through metrics calculation and statistical analysis.
- **Data visualization:** NumPy arrays can be easily integrated with libraries like Matplotlib to visualize data distributions and model predictions.

NumPy's versatility and performance make it a cornerstone of AI development, enabling researchers and practitioners to efficiently implement and experiment with various algorithms and models.

***

## Common NumPy Functions:

### [Array Creation](#array-creation-1)
1. np.array()
2. np.zeros()
3. np.zeros_like()
4. np.ones()
5. np.ones_like()
6. np.empty()
7. np.empty_like()
8. np.arange()
9. np.linspace()
10. np.eye()
11. np.copy()

### [Random Number Generation](#random-number-generation-1)
1. np.random.rand()
2. np.random.randn()
3. np.random.random()
4. np.random.random_sample()
5. np.random.randint()
6. np.random.choice()
7. np.random.shuffle()

### [Mathematical Functions](#Mathematical-Functions-1)
1. np.sum()
2. np.mean()
3. np.var()
4. np.std()
5. np.min()
6. np.max()
7. np.argmin()
8. np.argmax()
9. np.sin()
10. np.cos()
11. np.tan()
12. np.exp()
13. np.log()
14. np.sqrt()
15. np.square()
16. np.add()
17. np.subtract()
18. np.multiply()
19. np.divide()

### [Array Manipulation](#Array-Manipulation-1)
1. np.reshape()
2. np.ravel()
3. np.transpose()
4. np.concatenate()
5. np.split()
6. np.hstack()
7. np.vstack()

### [Linear Algebra](#Linear-Algebra-1)
1. np.dot()
2. np.matmul()
3. np.linalg.inv()
4. np.linalg.det()
5. np.linalg.eig()
6. np.linalg.solve()
7. np.linalg.lstsq()

### [Array Comparison and Boolean Operations](#Array-Comparison-and-Boolean-Operations-1)
1. np.equal()
2. np.not_equal()
3. np.logical_and()
4. np.logical_or()
5. np.logical_not()
6. np.all()
7. np.any()

### [Array Indexing and Slicing](#Array-Indexing-and-Slicing-1)
1. np.take()
2. np.put()
3. np.argmax()
4. np.argmin()
5. np.where()
6. np.extract()

### [Array Iteration](#Array-Iteration-1)
1. np.nditer()
2. np.ndindex()
3. np.ndenumerate()

### [Array Sorting and Searching](#Array-Sorting-and-Searching-1):
1. np.sort()
2. np.argsort()
3. np.searchsorted()

### [Array Set Operations](#Array-Set-Operations-1):
1. np.unique()
2. np.intersect1d()
3. np.union1d()
4. np.setdiff1d()

### [File Input and Output](#File-Input-and-Output-1):
1. np.loadtxt()
2. np.genfromtxt()
3. np.savetxt()

### [Array Reshaping and Resizing](#Array-Reshaping-and-Resizing-1):
1. np.resize()
2. np.expand_dims()
3. np.squeeze()
4. np.swapaxes()

### [Polynomial Functions](#Polynomial-Functions-1):
1. np.poly()
2. np.polyval()
3. np.polyfit()
4. np.roots()

### [Statistical Functions](#Statistical-Functions-1):
1. np.histogram()
2. np.bincount()
3. np.percentile()
4. np.corrcoef()

### [Fourier Transformations](#Fourier-Transformations-1):
1. np.fft.fft()
2. np.fft.ifft()
3. np.fft.fftfreq()

### [Other Utilities](#Other-Utilities-1):
1. +
2. -
3. *
4. /

***
***

### Array Creation:

#### 1- np.array()
Creating a NumPy array from a Python list or tuple.

##### Code:
```python
import numpy as np

my_list = [1, 2, 3, 4, 5]
my_array = np.array(my_list)
print(my_array)
```
##### Output:
```plaintext
[1 2 3 4 5]
```
---
#### 2- np.zeros()
Creating an array of zeros with a specified shape

##### Code:
```python
import numpy as np

zeros_array = np.zeros((3, 4))
print(zeros_array)
```
##### Output:
```plaintext
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
```
---
#### 3- np.zeros_like()
Creates an array of zeros with the same shape and type as a given array.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Create an array of zeros with the same shape as arr
zeros_like_arr = np.zeros_like(arr)
print(zeros_like_arr)
```
##### Output:
```plaintext
[[0 0 0]
 [0 0 0]]
```
---
#### 4- np.ones()
Creating an array of ones with a specified shape

##### Code:
```python
import numpy as np

ones_array = np.ones((4, 2))
print(ones_array)
```
##### Output:
```plaintext
[[1. 1.]
 [1. 1.]
 [1. 1.]
 [1. 1.]]
```
---


#### 5- np.ones_like()
Creates an array of ones with the same shape and type as a given array.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Create an array of ones with the same shape as arr
ones_like_arr = np.ones_like(arr)
print(ones_like_arr)
```
##### Output:
```plaintext
[[1 1 1]
 [1 1 1]]
```
---
#### 6- np.empty()
Creating an uninitialized array with a specified shape

##### Code:
```python
import numpy as np

empty_array = np.empty((3, 2))
print(empty_array)
```
##### Output:
```plaintext
[[1.04591558e-311 1.04591557e-311]
 [1.04591558e-311 1.04591558e-311]
 [1.04591558e-311 1.04591558e-311]]
```
---
#### 7- np.empty_like()
Creates an empty array with the same shape and type as a given array.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Create an empty (Random Values) array with the same shape as arr
empty_like_arr = np.empty_like(arr)
print(empty_like_arr)
```
##### Output:
```plaintext
[[-998669648      32761 -998665136]
 [     32761          0          0]]
```
---
#### 8- np.arange()
Creating an array of evenly spaced values within a given range

##### Code:
```python
import numpy as np

range_array = np.arange(0, 10, 2)  # Start, stop (exclusive), step
print(range_array)
```
##### Output:
```plaintext
[0 2 4 6 8]
```
---
#### 9- np.linspace()
Creating an array of evenly spaced numbers over a specified interval

##### Code:
```python
import numpy as np

linspace_array = np.linspace(0, 1, 5)  # Start, stop, number of points
print(linspace_array)
```
##### Output:
```plaintext
[0.  0.25  0.5  0.75  1.  ]
```

##### Code:
```python
import numpy as np

linspace_array = np.linspace(1, 5, 9)  # Start, stop, number of points
print(linspace_array)
```
##### Output:
```plaintext
[1.  1.5  2.  2.5  3.  3.5  4.  4.5  5. ]
```
---
#### 10- np.eye()
Creating a 2-D identity matrix (diagonal array of ones)

##### Code:
```python
import numpy as np

identity_matrix = np.eye(5)
print(identity_matrix)
```
##### Output:
```plaintext
[[1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]
```
***
#### 11. np.copy()
Creates a deep copy of an array.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Create a copy of arr
arr_copy = np.copy(arr)
print(arr_copy)
```
##### Output:
```plaintext
[[1 2 3]
 [4 5 6]]
```
***


| Function       | Description                                                | Syntax                                      |
|----------------|------------------------------------------------------------|---------------------------------------------|
| `np.array()`   | Create an array from a Python list or array-like object.   | `np.array(object, dtype=None, copy=True)`  |
| `np.zeros()`   | Create an array filled with zeros.                         | `np.zeros(shape, dtype=float, order='C')`  |
| `np.zeros_like()`   | Creates an array of zeros with the same shape and type as a given array. | `np.zeros_like(a, dtype=None, order='K', subok=True, shape=None)`  |
| `np.ones()`    | Create an array filled with ones.                          | `np.ones(shape, dtype=None, order='C')`    |
| `np.ones_like()`    | Creates an array of ones with the same shape and type as a given array. | `np.ones_like(a, dtype=None, order='K', subok=True, shape=None)`    |
| `np.empty()`   | Create an uninitialized array.                             | `np.empty(shape, dtype=float, order='C')`  |
| `np.empty_like()`   | Creates an empty array with the same shape and type as a given array. | `np.empty_like(a, dtype=None, order='K', subok=True, shape=None)`  |
| `np.arange()`  | Create an array with evenly spaced values within a range.  | `np.arange(start, stop, step, dtype=None)` |
| `np.linspace()`| Create an array with evenly spaced numbers over a specified interval.| `np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)`|
| `np.eye()`     | Create a 2-D identity matrix (diagonal array of ones).     | `np.eye(N, M=None, k=0, dtype=float)`      |
| `np.copy()`    | Creates a deep copy of an array.                           | `np.copy(a)`                               |

##### NOTE | 
`a`: This parameter represents the array-like input. It can be a NumPy array or any array-like object (e.g., list, tuple, etc.).

`dtype=None`: This parameter specifies the data type of the output array. If not provided, it defaults to the data type of the input a.

`order='K'`: This parameter is used to specify the memory layout of the array. It determines how elements of the array are stored in memory.
It can take two values:
'C' (default): This stands for C-style contiguous memory layout, also known as row-major order. In this layout, elements of each row are stored contiguously in memory.
'F': This stands for Fortran-style contiguous memory layout, also known as column-major order. In this layout, elements of each column are stored contiguously in memory.
'K': NumPy will keep the memory layout of the input array unchanged.

`subok=True`: This parameter specifies whether to allow subclasses of the input array.
It can take two values:
True (default): This allows subclasses of the input array a to be passed through. If the input array is a subclass, the output array will also be a subclass.
False         : This ensures that the output array will be of the base class (i.e., not a subclass), regardless of the input array's subclass status.
**Subclasses** in NumPy can provide additional functionality or behavior compared to the base array class. By default, NumPy functions preserve subclassing, but you can disable this behavior by setting subok to False.
Use **subok=True** when you want to retain subclassing in the output array. 
Use **subok=False** when you want to ensure that the output array is of the base class, regardless of the input array's subclass status.

`shape=None`: This parameter specifies the shape of the output array. If not provided, it defaults to the shape of the input array a.


***
***

### Random Number Generation:


#### 1- np.random.rand()
Generates random values in a given shape from a uniform distribution over [0, 1).

##### Code:
```python
import numpy as np

random_array = np.random.rand(2, 3, 2)
print(random_array)
```
##### Output:
```plaintext
[[[0.59439737 0.04203605]
  [0.89240467 0.95026019]]

 [[0.53163278 0.41530542]
  [0.87641794 0.48819283]]]
```
***
#### 2- np.random.randn()
Generate random values from the standard normal distribution

##### Code:
```python
import numpy as np

random_array = np.random.randn(2, 3)
print(random_array)
```
##### Output:
```plaintext
[[-0.79327957  0.88652359  1.30526375]
 [-1.03018159  1.3886636  -1.36125316]]
```
***
#### 3- np.random.random()
Generate a random float between 0 and 1

##### Code:
```python
import numpy as np

random_value = np.random.random((2, 3))
print(random_value)
```
##### Output:
```plaintext
[[0.42872708 0.7385717  0.16628425]
 [0.04095093 0.03574836 0.99828977]]
```
***
#### 4. np.random.random_sample()
Generate a random float between 0 and 1

##### Code:
```python
import numpy as np

random_value = np.random.random_sample((2, 2))
print(random_value)
```
##### Output:
```plaintext
[[0.76578554 0.94115212]
 [0.5710084  0.76463451]]
```
***
#### 5. np.random.randint()
Generates random integers from a specified low (inclusive) to high (exclusive) range.

##### Code1:
```python
import numpy as np

random_value = np.random.randint(1, 5)                # Generate a random integer number from 1 to 4 | NOTE : By default size = 1
print(random_value)
```
##### Output1:
```plaintext
3
```

##### Code2:
```python
import numpy as np

random_array = np.random.randint(1, 100, size=(2, 2)) # Generate random integer numbers from 1 to 99
print(random_array)
```
##### Output2:
```plaintext
[[38 56]
 [ 4 29]]
```

***
#### 6. np.random.choice()
Generates a random sample from a given 1-D array.

##### Code:
```python
import numpy as np

my_list = ['Mohamed', 'Abdalkader', 'Abdalsalam', 'Numpy', 'Python']
random_choice = np.random.choice(my_list)
print(random_choice)
```
##### Output:
```plaintext
Mohamed
```
***
#### 7. np.random.shuffle()
Shuffles the contents of a sequence in place.

##### Code:
```python
import numpy as np

my_array = np.arange(10)
print(f"Ordered Array  : {my_array}")

np.random.shuffle(my_array)
print(f"Shuffled Array : {my_array}")
```
##### Output:
```plaintext
Ordered Array  : [0 1 2 3 4 5 6 7 8 9]
Shuffled Array : [0 8 7 9 3 6 4 5 1 2]
```
***

| Function                      | Description                                                               | Syntax                                                             |
|-------------------------------|---------------------------------------------------------------------------|--------------------------------------------------------------------|
| `np.random.rand()`            | Generate random values in a given shape from a uniform distribution over [0, 1).| `np.random.rand(d0, d1, ..., dn)`                              |
| `np.random.randn()`           | Return a sample (or samples) from the "standard normal" distribution.     | `np.random.randn(d0, d1, ..., dn)`                             |
| `np.random.random()`         | Generate random floats in the half-open interval [0.0, 1.0).               | `np.random.random(size=None)`                                    |
| `np.random.random_sample()`   | Same as `np.random.random()`.                                             | `np.random.random_sample(size=None)`                             |
| `np.random.randint()`        | Return random integers from low (inclusive) to high (exclusive).          | `np.random.randint(low, high=None, size=None, dtype='l')`         |
| `np.random.choice()`         | Generate a random sample from a given 1-D array.                          | `np.random.choice(a, size=None, replace=True, p=None)`            |
| `np.random.shuffle()`        | Modify a sequence in-place by shuffling its contents.                     | `np.random.shuffle(x)`                                            |


***
***
### Mathematical Functions:

#### 1. np.sum()
Computes the sum of array elements over a given axis.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([[1, 2], [3, 4]])

# Compute the sum of all elements
total_sum = np.sum(arr)
print("Sum (Total)   =", total_sum)

# Compute the sum along the columns (axis=0)
col_sum = np.sum(arr, axis=0)
print("Sum (Columns) =", col_sum)

# Compute the sum along the rows (axis=1)
row_sum = np.sum(arr, axis=1)
print("Sum (Rows)    =", row_sum)
```
##### Output:
```plaintext
Sum (Total)   = 10
Sum (Columns) = [4 6]
Sum (Rows)    = [3 7]
```
***
#### 2. np.mean()
Computes the arithmetic mean along the specified axis.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([[1, 2], [3, 4]])

# Compute the mean of all elements
mean = np.mean(arr)
print("Mean (Total)   =", mean)

# Compute the mean along the columns (axis=0)
col_mean = np.mean(arr, axis=0)
print("Mean (Columns) =", col_mean)

# Compute the mean along the rows (axis=1)
row_mean = np.mean(arr, axis=1)
print("Mean (Rows)    =", row_mean)
```
##### Output:
```plaintext
Mean (Total)   = 2.5
Mean (Columns) = [2. 3.]
Mean (Rows)    = [1.5 3.5]
```
***
#### 3. np.var()
Computes the variance along the specified axis.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([[1, 2], [3, 4]])

# Compute the variance of all elements
total_variance = np.var(arr)
print("Variance (Total)   =", total_variance)

# Compute the variance along the columns (axis=0)
col_variance = np.var(arr, axis=0)
print("Variance (Columns) =", col_variance)

# Compute the variance along the rows (axis=1)
row_variance = np.var(arr, axis=1)
print("Variance (Rows)    =", row_variance)
```
##### Output:
```plaintext
Variance (Total)   = 1.25
Variance (Columns) = [1. 1.]
Variance (Rows)    = [0.25 0.25]
```
***
#### 4. np.std()
Computes the standard deviation along the specified axis.
 
##### Code:
```python
import numpy as np

# Example array
arr = np.array([[1, 2], [3, 4]])

# Compute the standard deviation of all elements
total_std = np.std(arr)
print("Standard Deviation (Total)   =", total_std)

# Compute the standard deviation along the columns (axis=0)
col_std = np.std(arr, axis=0)
print("Standard Deviation (Columns) =", col_std)

# Compute the standard deviation along the rows (axis=1)
row_std = np.std(arr, axis=1)
print("Standard Deviation (Rows)    =", row_std)
```
##### Output:
```plaintext
Standard Deviation (Total)   = 1.118033988749895
Standard Deviation (Columns) = [1. 1.]
Standard Deviation (Rows)    = [0.5 0.5]
```
***
#### 5. np.min()
Computes the minimum value along the specified axis.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([[1, 2], [3, 4]])

# Compute the minimum of all elements
total_min = np.min(arr)
print("Minimum (Total)   =", total_min)

# Compute the minimum along the columns (axis=0)
col_min = np.min(arr, axis=0)
print("Minimum (Columns) =", col_min)

# Compute the minimum along the rows (axis=1)
row_min = np.min(arr, axis=1)
print("Minimum (Rows)    =", row_min)
```
##### Output:
```plaintext
Minimum (Total)   = 1
Minimum (Columns) = [1 2]
Minimum (Rows)    = [1 3]
```
***
#### 6. np.max()
Computes the maximum value along the specified axis.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([[1, 2], [3, 4]])

# Compute the maximum of all elements
total_max = np.max(arr)
print("Maximum (Total)   =", total_max)

# Compute the maximum along the columns (axis=0)
col_max = np.max(arr, axis=0)
print("Maximum (Columns) =", col_max)

# Compute the maximum along the rows (axis=1)
row_max = np.max(arr, axis=1)
print("Maximum (Rows)    =", row_max)
```
##### Output:
```plaintext
Maximum (Total)   = 4
Maximum (Columns) = [3 4]
Maximum (Rows)    = [2 4]
```
***
#### 7. np.argmin()
Computes the indices of the minimum values along the specified axis.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([[1, 2], [3, 4]])

# Find the index of the minimum element
total_min_index = np.argmin(arr)
print("Index of Minimum (Total)     =", total_min_index)

# Find the index of the minimum along the columns (axis=0)
col_min_indices = np.argmin(arr, axis=0)
print("Indices of Minimum (Columns) =", col_min_indices)

# Find the index of the minimum along the rows (axis=1)
row_min_indices = np.argmin(arr, axis=1)
print("Indices of Minimum (Rows)    =", row_min_indices)
```
##### Output:
```plaintext
Index of Minimum (Total)     = 0
Indices of Minimum (Columns) = [0 0]
Indices of Minimum (Rows)    = [0 0]
```
***
#### 8. np.argmax()
Computes the indices of the maximum values along the specified axis.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([[1, 2], [3, 4]])

# Find the index of the maximum element
total_max_index = np.argmax(arr)
print("Index of Maximum (Total)   =", total_max_index)

# Find the index of the maximum along the columns (axis=0)
col_max_indices = np.argmax(arr, axis=0)
print("Indices of Maximum (Columns) =", col_max_indices)

# Find the index of the maximum along the rows (axis=1)
row_max_indices = np.argmax(arr, axis=1)
print("Indices of Maximum (Rows)    =", row_max_indices)
```
##### Output:
```plaintext
Index of Maximum (Total)   = 3
Indices of Maximum (Columns) = [1 1]
Indices of Maximum (Rows)    = [1 1]
```
***
#### 9. np.sin()
Computes the sine of the input array.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([[0, np.pi / 3], [np.pi / 4, np.pi / 6]])

# Compute the sine of all elements
sin_arr = np.sin(arr)
print(sin_arr)
```
##### Output:
```plaintext
[[0.         0.8660254 ]
 [0.70710678 0.5       ]]
```
***
#### 10. np.cos()
Computes the cosine of the input array.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([[0, np.pi / 3], [np.pi / 4, np.pi / 6]])

# Compute the cosine of all elements
cos_arr = np.cos(arr)
print(cos_arr)
```
##### Output:
```plaintext
[[1.         0.5       ]
 [0.70710678 0.8660254 ]]
```
***
#### 11. np.tan()
Computes the tangent of the input array.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([[0, np.pi/3], [np.pi/4, np.pi/6]])

# Compute the tangent of all elements
tan_arr = np.tan(arr)
print(tan_arr)
```
##### Output:
```plaintext
[[0.         1.73205081]
 [1.         0.57735027]]
```
***
#### 12. np.exp()
Computes the exponential of the input array.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([[1, 2], [3, 4]])

# Compute the exponentials of all elements
exp_arr = np.exp(arr)
print(exp_arr)
```
##### Output:
```plaintext
[[ 2.71828183  7.3890561 ]
 [20.08553692 54.59815003]]
```
***
#### 13. np.log() | np.log2() | np.log10()
`np.log()` Computes the natural logarithm of the input array.

`np.log()` Computes the base-2 logarithm of all elements in the array.

`np.log()` Computes the base-10 logarithm of all elements in the array.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([[np.e, np.e ** 3], [2, 8], [10, 1000]])

# Compute the natural logarithm of all elements
log_natural_arr = np.log(arr)
print("Natural logarithm:")
print(log_natural_arr)

# Compute the base-2 logarithm of all elements
log_base_2_arr = np.log2(arr)
print("\nLogarithm base 2:")
print(log_base_2_arr)

# Compute the base-10 logarithm of all elements
log_base_10_arr = np.log10(arr)
print("\nLogarithm base 10:")
print(log_base_10_arr)
```
##### Output:
```plaintext
Natural logarithm:
[[1.         3.        ]
 [0.69314718 2.07944154]
 [2.30258509 6.90775528]]

Logarithm base 2:
[[1.44269504 4.32808512]
 [1.         3.        ]
 [3.32192809 9.96578428]]

Logarithm base 10:
[[0.43429448 1.30288345]
 [0.30103    0.90308999]
 [1.         3.        ]]
```
***
#### 14. np.sqrt()
Computes the non-negative square root of the input array.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([[1, 4], [9, 16]])

# Compute the square root of all elements
sqrt_arr = np.sqrt(arr)
print(sqrt_arr)
```
##### Output:
```plaintext
[[1. 2.]
 [3. 4.]]
```
***
#### 15. np.square()
Computes the square of the input array. 

##### Code:
```python
import numpy as np

# Example array
arr = np.array([[1, 2], [3, 4]])

# Compute the square of all elements
square_arr = np.square(arr)
print(square_arr)
```
##### Output:
```plaintext
[[ 1  4]
 [ 9 16]]
```
***
#### 16. np.add()
Adds arguments element-wise.

##### Code:
```python
import numpy as np

# Example arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Element-wise addition
result = np.add(arr1, arr2)
print(result)
```
##### Output:
```plaintext
[5 7 9]
```
***
#### 17. np.subtract()
Subtracts arguments element-wise.

##### Code:
```python
import numpy as np

# Example arrays
arr1 = np.array([4, 5, 6])
arr2 = np.array([1, 2, 3])

# Element-wise subtraction
result = np.subtract(arr1, arr2)
print(result)
```
##### Output:
```plaintext
[3 3 3]
```
***
#### 18. np.multiply()
Multiplies arguments element-wise.

##### Code:
```python
import numpy as np

# Example arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Element-wise multiplication
result = np.multiply(arr1, arr2)
print(result)
```
##### Output:
```plaintext
[ 4 10 18]
```
***
#### 19. np.divide()
Returns a true division of the inputs, element-wise.

##### Code:
```python
import numpy as np

# Example arrays
arr1 = np.array([4, 6, 8])
arr2 = np.array([2, 3, 4])

# Element-wise division
result = np.divide(arr1, arr2)
print(result)
```
##### Output:
```plaintext
[2. 2. 2.]
```
***

| Function        | Description                                                       | Syntax                                      |
|-----------------|-------------------------------------------------------------------|---------------------------------------------|
| `np.sum()`      | Computes the sum of array elements over a given axis.             | `np.sum(a, axis=None, dtype=None, keepdims=False, initial=0)`  |
| `np.mean()`     | Computes the arithmetic mean along the specified axis.            | `np.mean(a, axis=None, dtype=None, out=None, keepdims=False)` |
| `np.var()`      | Computes the variance along the specified axis.                   | `np.var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False)` |
| `np.std()`      | Computes the standard deviation along the specified axis.         | `np.std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False)` |
| `np.min()`      | Computes the minimum value along the specified axis.              | `np.min(a, axis=None, out=None, keepdims=False, initial=<no value>, where=True)` |
| `np.max()`      | Computes the maximum value along the specified axis.              | `np.max(a, axis=None, out=None, keepdims=False, initial=<no value>, where=True)` |
| `np.argmin()`   | Computes the indices of the minimum values along the specified axis. | `np.argmin(a, axis=None, out=None)` |
| `np.argmax()`   | Computes the indices of the maximum values along the specified axis. | `np.argmax(a, axis=None, out=None)` |
| `np.sin()`      | Computes the sine of the input array.                             | `np.sin(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])` |
| `np.cos()`      | Computes the cosine of the input array.                           | `np.cos(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])` |
| `np.tan()`      | Computes the tangent of the input array.                          | `np.tan(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])` |
| `np.exp()`      | Computes the exponential of the input array.                      | `np.exp(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])` |
| `np.log()`      | Computes the natural logarithm of the input array.                | `np.log(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])` |
| `np.sqrt()`     | Computes the non-negative square root of the input array.         | `np.sqrt(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])` |
| `np.square()`   | Computes the square of the input array.                           | `np.square(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])` |
| `np.add()`      | Adds arguments element-wise.                                       | `np.add(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])` |
| `np.subtract()` | Subtracts arguments element-wise.                                  | `np.subtract(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])` |
| `np.multiply()` | Multiplies arguments element-wise.                                 | `np.multiply(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])` |
| `np.divide()`   | Returns a true division of the inputs, element-wise.               | `np.divide(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])` |
| `np.dot()`      | Dot product of two arrays.                                        | `np.dot(a, b, out=None)`                   |

***
***
### Array Manipulation:

#### 1. np.reshape()
Reshapes an array without changing its data.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# Reshape the array to a 3x4 matrix
reshaped_arr = np.reshape(arr, (3, 4))
print(reshaped_arr)
```
##### Output:
```plaintext
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]
```
***
#### 2. np.ravel()
Returns a flattened array.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Flatten the array
flattened_arr = np.ravel(arr)
print(flattened_arr)
```
##### Output:
```plaintext
[1 2 3 4 5 6]
```
***
#### 3. np.transpose()
Permute the dimensions of an array.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Transpose the array
transposed_arr = np.transpose(arr)
print(transposed_arr)
```
##### Output:
```plaintext
[[1 4]
 [2 5]
 [3 6]]
```
***
#### 4. np.concatenate()
Joins a sequence of arrays along an existing axis.

##### Code:
```python
import numpy as np

# Example arrays
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6]])

# Concatenate the arrays along the rows (axis=0)
concatenated_arr = np.concatenate((arr1, arr2), axis=0)
print("Concatenated along rows:")
print(concatenated_arr)

# Example arrays for concatenating along columns
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5], [6]])

# Concatenate the arrays along the columns (axis=1)
concatenated_arr = np.concatenate((arr1, arr2), axis=1)
print("\nConcatenated along columns:")
print(concatenated_arr)
```
##### Output:
```plaintext
Concatenated along rows:
[[1 2]
 [3 4]
 [5 6]]

Concatenated along columns:
[[1 2 5]
 [3 4 6]]
```
***
#### 5. np.split()
Splits an array into multiple sub-arrays.

##### Code:
```python
import numpy as np

# Example array
arr = np.arange(12)

# Split the array into three sub-arrays
split_arr = np.split(arr, 4)
print(split_arr)
```
##### Output:
```plaintext
[array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8]), array([ 9, 10, 11])]
```
***
#### 6. np.hstack()
Stacks arrays in sequence horizontally (column-wise).

##### Code:
```python
import numpy as np

# Example arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Stack the arrays horizontally
stacked_arr = np.hstack((arr1, arr2))
print(stacked_arr)
```
##### Output:
```plaintext
[1 2 3 4 5 6]
```
***
#### 7. np.vstack()
Stacks arrays in sequence vertically (row-wise).

##### Code:
```python
import numpy as np

# Example arrays
arr1 = np.array([[1], [2], [3]])
arr2 = np.array([[4], [5], [6]])

# Stack the arrays vertically
stacked_arr = np.vstack((arr1, arr2))
print(stacked_arr)
```
##### Output:
```plaintext
[[1]
 [2]
 [3]
 [4]
 [5]
 [6]]
```
***
***
### Linear Algebra:
   
#### 1. np.dot()
Dot product of two arrays.

##### Code:
```python
import numpy as np

# Example arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Dot product
result = np.dot(arr1, arr2)
print(result)
```
##### Output:
```plaintext
32
```
***
#### 2. np.matmul()
Matrix product of two arrays.

##### Code:
```python
import numpy as np

# Example arrays
arr1 = np.array([[1, 2],
                 [3, 4]])

arr2 = np.array([[5, 6],
                 [7, 8]])

# Matrix multiplication
result = np.matmul(arr1, arr2)
print(result)
```
##### Output:
```plaintext
[[19 22]
 [43 50]]
```
***
#### 3. np.linalg.inv()
Computes the inverse of a matrix.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([[1, 2], [3, 4]])

# Compute the inverse of the matrix
inverse_matrix = np.linalg.inv(arr)
print(inverse_matrix)
```
##### Output:
```plaintext
[[-2.   1. ]
 [ 1.5 -0.5]]
```
***
#### 4. np.linalg.det()
Computes the determinant of a square matrix.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([[1, 2], [3, 4]])

# Compute the determinant of the matrix
determinant = np.linalg.det(arr)
print(determinant)
```
##### Output:
```plaintext
-2
```
***
#### 5. np.linalg.eig()
Computes the eigenvalues and right eigenvectors of a square array.

Eigenvalues: Eigenvalues are scalar values that represent how a linear transformation, represented by a matrix, stretches or contracts vectors. For a square matrix A, an eigenvalue λ and its corresponding eigenvector v satisfy the equation Av = λv. Each eigenvalue describes a scaling factor for its corresponding eigenvector.

Eigenvectors: Eigenvectors are non-zero vectors that, when transformed by a matrix, only change in scale (magnitude) and not in direction. They represent the directions along which the linear transformation represented by the matrix has a simple effect.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([[1, 2], [3, 4]])

# Compute the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(arr)
print("Eigenvalues:", eigenvalues, sep="\n ", end="\n\n")
print("Eigenvectors:", eigenvectors, sep="\n")
```
##### Output:
```plaintext
Eigenvalues:
 [-0.37228132  5.37228132]

Eigenvectors:
[[-0.82456484 -0.41597356]
 [ 0.56576746 -0.90937671]]
```
***
#### 6. np.linalg.solve()
Solves a linear matrix equation.

##### Code:
```python
import numpy as np

# Example arrays
A = np.array([[2, 1], [1, 1]])
b = np.array([1, 1])

# Solve the linear equation Ax = b
x = np.linalg.solve(A, b)
print(x)
```
##### Output:
```plaintext
[0. 1.]
```
##### Explination:
`A` represents the coefficient matrix of the system of linear equations.
`b` represents the constant terms on the right-hand side of the equations.

`np.linalg.solve(A, b)` solves the equation `Ax = b`, where `A` is the coefficient matrix and `b` is the constant vector.
The function returns the solution `vector x`, which satisfies the equation `Ax = b`.

```plaintext
2x + 1y = 1
1x + 1y = 1


| 2  1 |   | x |   | 1 |
|      | * |   | = |   |
| 1  1 |   | y |   | 1 |


x=0, y=1
```
***
#### 7. np.linalg.lstsq()
Computes the least-squares solution to a linear matrix equation.

##### Code:
```python
import numpy as np

# Example arrays
A = np.array([[0, 1], [1, 1], [2, 1]])
b = np.array([0, 1, 2])

# Compute the least-squares solution
solution = np.linalg.lstsq(A, b, rcond=None)
print(solution)
```
##### Output:
```plaintext
(array([-0.33333333,  0.83333333]), array([], dtype=float64), 2, array([2.61803399, 1.        ]))
```
***
***
### Array Comparison and Boolean Operations:

#### 1. np.equal()
Tests element-wise equality of two arrays.

##### Code:
```python
import numpy as np

# Example arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([1, 2, 3])

# Test for equality
result = np.equal(arr1, arr2)
print(result)
```
##### Output:
```plaintext
[True True True]
```
***
#### 2. np.not_equal()
Tests element-wise inequality of two arrays.

##### Code:
```python
import numpy as np

# Example arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([1, 0, 3])

# Test for inequality
result = np.not_equal(arr1, arr2)
print(result)
```
##### Output:
```plaintext
[False True False]
```
***
#### 3. np.logical_and()
Computes the truth value of x1 AND x2 element-wise.

##### Code:
```python
import numpy as np

# Example arrays
arr1 = np.array([True, True, False])
arr2 = np.array([True, False, False])

# Logical AND operation
result = np.logical_and(arr1, arr2)
print(result)
```
##### Output:
```plaintext
[True False False]
```
***
#### 4. np.logical_or()
Computes the truth value of x1 OR x2 element-wise.

##### Code:
```python
import numpy as np

# Example arrays
arr1 = np.array([True, True, False])
arr2 = np.array([True, False, False])

# Logical OR operation
result = np.logical_or(arr1, arr2)
print(result)
```
##### Output:
```plaintext
[True  True False]
```
***
#### 5. np.logical_not()
Computes the truth value of NOT x element-wise.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([True, False, True])

# Logical NOT operation
result = np.logical_not(arr)
print(result)
```
##### Output:
```plaintext
[False True False]
```
***
#### 6. np.all()
Tests whether all array elements along a given axis evaluate to True.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([True, True, True])

# Test if all elements are True
result = np.all(arr)
print(result)
```
##### Output:
```plaintext
True
```
***
#### 7. np.any()
Tests whether any array element along a given axis evaluates to True.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([False, False, True])

# Test if any element is True
result = np.any(arr)
print(result)
```
##### Output:
```plaintext
True
```
***
***

### Array Indexing and Slicing:

#### 1. np.take()
Returns elements from an array along an axis.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([4, 3, 5, 7])

# Take elements at specified indices
indices = [0, 2]
result = np.take(arr, indices)
print(result)
```
##### Output:
```plaintext
[4 5]
```
***
#### 2. np.put()
Replaces specified elements of an array with given values.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([1, 2, 3, 4])

# Replace elements at specified indices
indices = [0, 2]
values = [10, 20]
np.put(arr, indices, values)
print(arr)
```
##### Output:
```plaintext
[10 2 20 4]
```
***
#### 3. np.argmax()
Returns the indices of the maximum values along an axis.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([3, 1, 5, 2, 7])

# Find the index of the maximum value
index = np.argmax(arr)
print(index)
```
##### Output:
```plaintext
4
```
***
#### 4. np.argmin()
Returns the indices of the minimum values along an axis.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([3, 1, 5, 2, 7])

# Find the index of the minimum value
index = np.argmin(arr)
print(index)
```
##### Output:
```plaintext
1
```
***
#### 5. np.where()
Return elements chosen from x or y depending on condition.

##### Code:
```python
import numpy as np

# Example arrays
condition = np.array([True, False, True])
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

# Choose elements based on condition
result = np.where(condition, x, y)
print(result)
```
##### Output:
```plaintext
[1 5 3]
```
***
#### 6. np.extract()
Return the elements of an array that satisfy some condition.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([1, 2, 3, 4, 5])

# Extract elements greater than 2
condition = arr > 2
result = np.extract(condition, arr)
print(result)
```
##### Output:
```plaintext
[3 4 5]
```
***
***

### Array Iteration:

#### 1. np.nditer()
Iterates over an array applying operation for each element.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([[1, 2], [3, 4]])

# Iterate over the array using nditer
for x in np.nditer(arr):
    print(x)
```
##### Output:
```plaintext
1
2
3
4
```
***
#### 2. np.ndindex()
Iterates over multi-dimensional index arrays.

##### Code1:
```python
import numpy as np

# Example array shape
shape = (2, 2)

# Iterate over indices using ndindex
for index in np.ndindex(shape):
    print(index)
```
##### Output1:
```plaintext
(0, 0)
(0, 1)
(1, 0)
(1, 1)
```

##### Code2:
```python
import numpy as np

# Example array shape
shape = (2, 2, 2)

# Iterate over indices using ndindex
for index in np.ndindex(shape):
    print(index)
```
##### Output2:
```plaintext
(0, 0, 0)
(0, 0, 1)
(0, 1, 0)
(0, 1, 1)
(1, 0, 0)
(1, 0, 1)
(1, 1, 0)
(1, 1, 1)
```
***
#### 3. np.ndenumerate()
Iterates over an array and yields index and value of each element.

##### Code1:
```python
import numpy as np

# Example array
arr = np.array([[1, 2],
                [3, 4]])

# Enumerate over the array using ndenumerate
for index, value in np.ndenumerate(arr):
    print("Index:", index, "Value:", value)
```
##### Output1:
```plaintext
Index: (0, 0) Value: 1
Index: (0, 1) Value: 2
Index: (1, 0) Value: 3
Index: (1, 1) Value: 4
```

##### Code2:
```python
import numpy as np

# Example array
arr = np.array([[[1, 2], [3, 4]],
                [[5, 6], [7, 8]]])

# Enumerate over the array using ndenumerate
for index, value in np.ndenumerate(arr):
    print("Index:", index, "Value:", value)
```
##### Output2:
```plaintext
Index: (0, 0, 0) Value: 1
Index: (0, 0, 1) Value: 2
Index: (0, 1, 0) Value: 3
Index: (0, 1, 1) Value: 4
Index: (1, 0, 0) Value: 5
Index: (1, 0, 1) Value: 6
Index: (1, 1, 0) Value: 7
Index: (1, 1, 1) Value: 8
```

***
***

### Array Sorting and Searching:

#### 1. np.sort()
Returns a sorted copy of an array.

##### Code1:
```python
import numpy as np

# Example array
arr = np.array([3, 1, 4, 2])

# Sort the array
sorted_arr = np.sort(arr)
print(sorted_arr)
```
##### Output1:
```plaintext
[1 2 3 4]
```

##### Code2:
```python
import numpy as np

# Example array
arr = np.array([[3, 1], 
                [4, 2]])

# Sort the array
sorted_arr = np.sort(arr)
print(sorted_arr)
```
##### Output2:
```plaintext
[[1 3]
 [2 4]]
```

***
#### 2. np.argsort()
Returns the indices that would sort an array.

##### Code1:
```python
import numpy as np

# Example array
arr = np.array([3, 1, 4, 2])

# Get the indices that would sort the array
indices = np.argsort(arr)
print(indices)
```
##### Output1:
```plaintext
[1 3 0 2]
```

##### Code2:
```python
import numpy as np

# Example array
arr = np.array([[3, 1],
                [4, 2]])

# Get the indices that would sort the array
indices = np.argsort(arr)
print(indices)
```
##### Output2:
```plaintext
[[1 0]
 [1 0]]
```

***
#### 3. np.searchsorted()
Finds the indices into a sorted array where elements should be inserted to maintain order.

##### Code1:
```python
import numpy as np

# Example sorted array
arr = np.array([1, 3, 5, 7, 9])

# Find indices where 4 should be inserted to maintain order
indices = np.searchsorted(arr, 4)
print(indices)
```
##### Output1:
```plaintext
2
```

##### Code2:
```python
import numpy as np

# Example sorted array
arr = np.array([1, 3, 5, 7, 9])

# Find indices where 10 should be inserted to maintain order
indices = np.searchsorted(arr, 10)
print(indices)
```
##### Output2:
```plaintext
5
```

***
***

### Array Set Operations:

#### 1. np.unique()
Finds the unique elements of an array.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([1, 2, 1, 3, 4, 2, 5])

# Find unique elements
unique_elements = np.unique(arr)
print(unique_elements)
```
##### Output:
```plaintext
[1 2 3 4 5]
```
***
#### 2. np.intersect1d()
Finds the intersection of two arrays.

##### Code:
```python
import numpy as np

# Example arrays
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([3, 4, 5, 6])

# Find intersection
intersection = np.intersect1d(arr1, arr2)
print(intersection)
```
##### Output:
```plaintext
[3 4]
```
***
#### 3. np.union1d()
Finds the union of two arrays.

##### Code:
```python
import numpy as np

# Example arrays
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([3, 4, 5, 6])

# Find union
union = np.union1d(arr1, arr2)
print(union)
```
##### Output:
```plaintext
[1 2 3 4 5 6]
```
***
#### 4. np.setdiff1d()
Finds the set difference of two arrays.

##### Code1:
```python
import numpy as np

# Example arrays
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([3, 4, 5, 6])

# Find set difference
difference = np.setdiff1d(arr1, arr2)
print(difference)
```
##### Output1:
```plaintext
[1 2]
```

##### Code2:
```python
import numpy as np

# Example arrays
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([3, 4, 5, 6])

# Find set difference
difference = np.setdiff1d(arr2, arr1)
print(difference)
```
##### Output2:
```plaintext
[5 6]
```

***
***

### File Input and Output:

#### 1. np.loadtxt()
Loads data from a text file.

##### Code:
```python
import numpy as np

# Load data from a text file
data = np.loadtxt('data.txt')
print(data)
```
##### Output:
```plaintext
Output depends on the content of the 'data.txt' file.
```
***
#### 2. np.genfromtxt()
Loads data from a text file with more options than np.loadtxt().

##### Code:
```python
import numpy as np

# Load data from a text file with specific options
data = np.genfromtxt('data.txt', delimiter=',', skip_header=1)
print(data)
```
##### Output:
```plaintext
Output depends on the content of the 'data.txt' file.
```
***
#### 3. np.savetxt()
Saves an array to a text file.

##### Code:
```python
import numpy as np

# Example array
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Save array to a text file
np.savetxt('output.txt', arr, delimiter=',')
```
##### Output:
```plaintext
No explicit output, The array is saved to 'output.txt' file.
```
***
***

### Array Reshaping and Resizing:

#### 1. np.resize()
DESC

##### Code:
```python

```
##### Output:
```plaintext

```
***
#### 2. np.expand_dims()
DESC

##### Code:
```python

```
##### Output:
```plaintext

```
***
#### 3. np.squeeze()
DESC

##### Code:
```python

```
##### Output:
```plaintext

```
***
#### 4. np.swapaxes()
DESC

##### Code:
```python

```
##### Output:
```plaintext

```
***
***

### Polynomial Functions:

#### 1. np.poly()
DESC

##### Code:
```python

```
##### Output:
```plaintext

```
***
#### 2. np.polyval()
DESC

##### Code:
```python

```
##### Output:
```plaintext

```
***
#### 3. np.polyfit()
DESC

##### Code:
```python

```
##### Output:
```plaintext

```
***
#### 4. np.roots()
DESC

##### Code:
```python

```
##### Output:
```plaintext

```
***
***

### Statistical Functions:

#### 1. np.histogram()
DESC

##### Code:
```python

```
##### Output:
```plaintext

```
***
#### 2. np.bincount()
DESC

##### Code:
```python

```
##### Output:
```plaintext

```
***
#### 3. np.percentile()
DESC

##### Code:
```python

```
##### Output:
```plaintext

```
***
#### 4. np.corrcoef()
DESC

##### Code:
```python

```
##### Output:
```plaintext

```
***
***

### Fourier Transformations:

#### 1. np.fft.fft()
DESC

##### Code:
```python

```
##### Output:
```plaintext

```
***
#### 2. np.fft.ifft()
DESC

##### Code:
```python

```
##### Output:
```plaintext

```
***
#### 3. np.fft.fftfreq()
DESC

##### Code:
```python

```
##### Output:
```plaintext

```
***
***

### Other Utilities:

#### 1. +
DESC

##### Code:
```python

```
##### Output:
```plaintext

```
***
#### 2. -
DESC

##### Code:
```python

```
##### Output:
```plaintext

```
***
#### 3. *
DESC

##### Code:
```python

```
##### Output:
```plaintext

```
***
#### 4. /
DESC

##### Code:
```python

```
##### Output:
```plaintext

```
***
