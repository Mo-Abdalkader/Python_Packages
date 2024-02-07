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

### 1- Array Creation:
1. np.array()
2. np.zeros()
3. np.ones()
4. np.empty()
5. np.arange()
6. np.linspace()
7. np.eye()

### 2- Random Number Generation:
1. np.random.rand()
2. np.random.randn()
3. np.random.randint()
4. np.random.choice()
5. np.random.shuffle()

### 3- Mathematical Functions:
1. np.sum()
2. np.mean()
3. np.std()
4. np.var()
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

### 4- Array Manipulation:
1. np.reshape()
2. np.ravel()
3. np.transpose()
4. np.concatenate()
5. np.split()
6. np.hstack()
7. np.vstack()

### 5- Linear Algebra:
1. np.dot()
2. np.matmul()
3. np.linalg.inv()
4. np.linalg.det()
5. np.linalg.eig()
6. np.linalg.solve()
7. np.linalg.lstsq()

### 6- Array Comparison and Boolean Operations:
1. np.equal()
2. np.not_equal()
3. np.logical_and()
4. np.logical_or()
5. np.logical_not()
6. np.all()
7. np.any()

### 7- Array Indexing and Slicing:
1. np.take()
2. np.put()
3. np.argmax()
4. np.argmin()
5. np.where()
6. np.extract()

### 8- Array Iteration:
1. np.nditer()
2. np.ndenumerate()
3. np.ndindex()

### 9- Array Sorting and Searching:
1. np.sort()
2. np.argsort()
3. np.searchsorted()

### 10- Array Set Operations:
1. np.unique()
2. np.intersect1d()
3. np.union1d()
4. np.setdiff1d()

### 11- File Input and Output:
1. np.loadtxt()
2. np.genfromtxt()
3. np.savetxt()

### 12- Array Reshaping and Resizing:
1. np.resize()
2. np.expand_dims()
3. np.squeeze()
4. np.swapaxes()

### 13- Polynomial Functions:
1. np.poly()
2. np.polyval()
3. np.polyfit()
4. np.roots()

### 14- Statistical Functions:
1. np.histogram()
2. np.bincount()
3. np.percentile()
4. np.corrcoef()

### 15- Fourier Transformations:
1. np.fft.fft()
2. np.fft.ifft()
3. np.fft.fftfreq()

### 16- Other Utilities:
1. np.zeros_like()
2. np.ones_like()
3. np.empty_like()
4. np.copy()



***
***

### 1- Array Creation:

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
