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

### [Array Creation:](#Array Creation)
1. np.array()
2. np.zeros()
3. np.ones()
4. np.empty()
5. np.arange()
6. np.linspace()
7. np.eye()

### [Random Number Generation:](#Random Number Generation)
1. np.random.rand()
2. np.random.randn()
3. np.random.random()
4. np.random.random_sample()
5. np.random.randint()
6. np.random.choice()
7. np.random.shuffle()

### [Mathematical Functions:](#Mathematical Functions)
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
20. np.dot()

### Array Manipulation:
1. np.reshape()
2. np.ravel()
3. np.transpose()
4. np.concatenate()
5. np.split()
6. np.hstack()
7. np.vstack()

### Linear Algebra:
1. np.dot()
2. np.matmul()
3. np.linalg.inv()
4. np.linalg.det()
5. np.linalg.eig()
6. np.linalg.solve()
7. np.linalg.lstsq()

### Array Comparison and Boolean Operations:
1. np.equal()
2. np.not_equal()
3. np.logical_and()
4. np.logical_or()
5. np.logical_not()
6. np.all()
7. np.any()

### Array Indexing and Slicing:
1. np.take()
2. np.put()
3. np.argmax()
4. np.argmin()
5. np.where()
6. np.extract()

### Array Iteration:
1. np.nditer()
2. np.ndenumerate()
3. np.ndindex()

### Array Sorting and Searching:
1. np.sort()
2. np.argsort()
3. np.searchsorted()

### Array Set Operations:
1. np.unique()
2. np.intersect1d()
3. np.union1d()
4. np.setdiff1d()

### File Input and Output:
1. np.loadtxt()
2. np.genfromtxt()
3. np.savetxt()

### Array Reshaping and Resizing:
1. np.resize()
2. np.expand_dims()
3. np.squeeze()
4. np.swapaxes()

### Polynomial Functions:
1. np.poly()
2. np.polyval()
3. np.polyfit()
4. np.roots()

### Statistical Functions:
1. np.histogram()
2. np.bincount()
3. np.percentile()
4. np.corrcoef()

### Fourier Transformations:
1. np.fft.fft()
2. np.fft.ifft()
3. np.fft.fftfreq()

### Other Utilities:
1. np.zeros_like()
2. np.ones_like()
3. np.empty_like()
4. np.copy()



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
#### 3- np.ones()
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
#### 4- np.empty()
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
#### 5- np.arange()
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
#### 6- np.linspace()
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
#### 7- np.eye()
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

| Function       | Description                                                | Syntax                                      |
|----------------|------------------------------------------------------------|---------------------------------------------|
| `np.array()`   | Create an array from a Python list or array-like object.   | `np.array(object, dtype=None, copy=True)`  |
| `np.zeros()`   | Create an array filled with zeros.                         | `np.zeros(shape, dtype=float, order='C')`  |
| `np.ones()`    | Create an array filled with ones.                          | `np.ones(shape, dtype=None, order='C')`    |
| `np.empty()`   | Create an uninitialized array.                             | `np.empty(shape, dtype=float, order='C')`  |
| `np.arange()`  | Create an array with evenly spaced values within a range.  | `np.arange(start, stop, step, dtype=None)` |
| `np.linspace()`| Create an array with evenly spaced numbers over a specified interval.| `np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)`|
| `np.eye()`     | Create a 2-D identity matrix (diagonal array of ones).     | `np.eye(N, M=None, k=0, dtype=float)`      |

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
print("Total Sum   =", total_sum)

# Compute the sum along the rows (axis=0)
row_sum = np.sum(arr, axis=0)
print("Rows Sum    =", row_sum)

# Compute the sum along the columns (axis=1)
col_sum = np.sum(arr, axis=1)
print("Columns Sum =", col_sum)
```
##### Output:
```plaintext
Total Sum   = 10
Rows Sum    = [4 6]
Columns Sum = [3 7]
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
print("Total mean   =", mean)

# Compute the mean along the rows (axis=0)
row_mean = np.mean(arr, axis=0)
print("Rows mean    =", row_mean)

# Compute the mean along the columns (axis=1)
col_mean = np.mean(arr, axis=1)
print("Columns mean =", col_mean)
```
##### Output:
```plaintext
Total mean   = 2.5
Rows mean    = [2. 3.]
Columns mean = [1.5 3.5]
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
mean = np.var(arr)
print("Total variance   =", mean)

# Compute the variance along the rows (axis=0)
row_mean = np.var(arr, axis=0)
print("Rows variance    =", row_mean)

# Compute the variance along the columns (axis=1)
col_mean = np.var(arr, axis=1)
print("Columns variance =", col_mean)
```
##### Output:
```plaintext
Total variance   = 1.25
Rows variance    = [1. 1.]
Columns variance = [0.25 0.25]
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
mean = np.std(arr)
print("Total standard deviation   =", mean)

# Compute the standard deviation along the rows (axis=0)
row_mean = np.std(arr, axis=0)
print("Rows standard deviation    =", row_mean)

# Compute the standard deviation along the columns (axis=1)
col_mean = np.std(arr, axis=1)
print("Columns standard deviation =", col_mean)
```
##### Output:
```plaintext
Total standard deviation   = 1.118033988749895
Rows standard deviation    = [1. 1.]
Columns standard deviation = [0.5 0.5]
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
mean = np.min(arr)
print("Total Minimum   =", mean)

# Compute the minimum along the rows (axis=0)
row_mean = np.min(arr, axis=0)
print("Rows Minimum    =", row_mean)

# Compute the minimum along the columns (axis=1)
col_mean = np.min(arr, axis=1)
print("Columns Minimum =", col_mean)
```
##### Output:
```plaintext
Total Minimum   = 1
Rows Minimum    = [1 2]
Columns Minimum = [1 3]
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
mean = np.max(arr)
print("Total Maximum   =", mean)

# Compute the maximum along the rows (axis=0)
row_mean = np.max(arr, axis=0)
print("Rows Maximum    =", row_mean)

# Compute the maximum along the columns (axis=1)
col_mean = np.max(arr, axis=1)
print("Columns Maximum =", col_mean)
```
##### Output:
```plaintext
Total Maximum   = 4
Rows Maximum    = [3 4]
Columns Maximum = [2 4]
```
***
#### 7. np.argmin()
Computes the indices of the minimum values along the specified axis.

##### Code:
```python

```
##### Output:
```plaintext

```
***
#### 8. np.argmax()
Computes the indices of the maximum values along the specified axis.

##### Code:
```python

```
##### Output:
```plaintext

```
***
#### 9. np.sin()
Computes the sine of the input array.

##### Code:
```python

```
##### Output:
```plaintext

```
***
#### 10. np.cos()
Computes the cosine of the input array.

##### Code:
```python

```
##### Output:
```plaintext

```
***
#### 11. np.tan()
Computes the tangent of the input array.

##### Code:
```python

```
##### Output:
```plaintext

```
***
#### 12. np.exp()
Computes the exponential of the input array.

##### Code:
```python

```
##### Output:
```plaintext

```
***
#### 13. np.log()
Computes the natural logarithm of the input array.

##### Code:
```python

```
##### Output:
```plaintext

```
***
#### 14. np.sqrt()
Computes the non-negative square root of the input array.

##### Code:
```python

```
##### Output:
```plaintext

```
***
#### 15. np.square()
Computes the square of the input array. 

##### Code:
```python

```
##### Output:
```plaintext

```
***
#### 16. np.add()
Adds arguments element-wise.

##### Code:
```python

```
##### Output:
```plaintext

```
***
#### 17. np.subtract()
Subtracts arguments element-wise.

##### Code:
```python

```
##### Output:
```plaintext

```
***
#### 18. np.multiply()
Multiplies arguments element-wise.

##### Code:
```python

```
##### Output:
```plaintext

```
***
#### 19. np.divide()
Returns a true division of the inputs, element-wise.

##### Code:
```python

```
##### Output:
```plaintext

```
***
#### 20. np.dot()
Dot product of two arrays.

##### Code:
```python

```
##### Output:
```plaintext

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

#### 

##### Code:
```python

```
##### Output:
```plaintext

```





