# Numpy



NumPy Reference https://docs.scipy.org/doc/numpy/reference/index.html

### numpy.ndarray
An array object represents a multidimensional, homogeneous array of fixed-size items.
Examples


```
np.ndarray(shape=(2,2), dtype=float, order='F')
ndarray.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)
```


Copy of the array, cast to a specified type.

### numpy.reshape


```
numpy.reshape(a, newshape, order='C')[source]
```


Gives a new shape to an array without changing its data.


```
>>> a = np.arange(6).reshape((3, 2))
>>> np.reshape(a, (3,-1))       # the unspecified value is inferred to be 2
array([[1, 2],
       [3, 4],
       [5, 6]])

```


### Routines
**Mathematical functions**

```
numpy.multiply(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) = <ufunc 'multiply'>
```
Multiply arguments element-wise.

```
numpy.log(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) = <ufunc 'log'>
```
Natural logarithm, element-wise.
The natural logarithm log is the inverse of the exponential function, so that log(exp(x)) = x. The natural logarithm is logarithm in base e.

```

numpy.sum(a, axis=None, dtype=None, out=None, keepdims=<class numpy._globals._NoValue>)
```
Sum of array elements over a given axis.

**Array manipulation routines**


```
numpy.concatenate((a1, a2, ...), axis=0)
```


Join a sequence of arrays along an existing axis.

**Array creation routines**


```
numpy.frombuffer(buffer, dtype=float, count=-1, offset=0)
```
Interpret a buffer as a 1-dimensional array.




### Logic functions
**Comparison**


```
numpy.array_equal(a1, a2)[source]
```


True if two arrays have the same shape and elements, False otherwise.

### Statistics


```
numpy.mean(a, axis=None, dtype=None, out=None, keepdims=<class numpy._globals._NoValue>)[source]
```



```

numpy.std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=<class numpy._globals._NoValue>)[source]
```



### Random sampling (numpy.random)
**Simple random data**


```
numpy.random.choice(a, size=None, replace=True, p=None)
```


Generates a random sample from a given 1-D array

**Permutations**


```
numpy.random.shuffle(x)
```


Modify a sequence in-place by shuffling its contents.


```

numpy.random.permutation(x)
```


Randomly permute a sequence, or return a permuted range.
If x is a multi-dimensional array, it is only shuffled along its first index.

### Indexing routines


```
numpy.where(condition[, x, y])
```


Return elements, either from x or y, depending on condition.








