# Basic Python Knowledge


https://docs.python.org/3.0/library/index.html

### Internet Protocols and Support



```
urllib.request.urlretrieve(url[, filename[, reporthook[, data]]])
```


"The second argument, if present, specifies the file location to copy to (if absent, the location will be a tempfile with a generated name). The third argument, if present, is a hook function that will be called once on establishment of the network connection and once after each block read thereafter. The hook will be passed three arguments; a count of blocks transferred so far, a block size in bytes, and the total size of the file. The third argument may be -1 on older FTP servers which do not return a file size in response to a retrieval request."

Return a tuple (filename, headers) where filename is the local file name under which the object can be found, and headers is whatever the info() method of the object returned by urlopen() returned (for a remote object, possibly cached).

### Generic Operating System Services
**os — Miscellaneous operating system interfaces**


```
os.stat(path)
```


Perform a stat system call on the given path. 


```
os.listdir(path)
```


Return a list containing the names of the entries in the directory given by path.

### File and Directory Access
**os.path — Common pathname manipulations**


```
os.path.exists(path)
```


Return True if path refers to an existing path. Returns False for broken symbolic links.


```
os.path.isdir(path)
```


Return True if path is an existing directory.


```
os.path.join(path1[, path2[, ...]])
```


Join one or more path components intelligently.


```
os.path.splitext(path)
```


Split the pathname path into a pair (root, ext) such that root + ext == path, and ext is empty 
or begins with a period and contains at most one period. Leading periods on the basename are ignored; 
splitext('.cshrc') returns ('.cshrc', '').


```

sys.stdout.flush() 
```


https://stackoverflow.com/questions/10019456/usage-of-sys-stdout-flush-method
Python's standard out is buffered (meaning that it collects some of the data "written" to standard out before it writes it to the terminal). Calling sys.stdout.flush() forces it to "flush" the buffer, meaning that it will write everything in the buffer to the terminal, even if normally it would wait before doing so.

### Data Compression and Archiving
**tarfile — Read and write tar archive files**


```
tarfile.open(name=None, mode=’r’, fileobj=None, bufsize=10240, **kwargs)
```


Return a TarFile object for the pathname name. 


```
TarFile.extractall(path=”.”, members=None, *, numeric_owner=False)
```


Extract all members from the archive to the current working directory or directory path.

Example:


```
import tarfile
tar = tarfile.open(filename)
sys.stdout.flush()           
tar.extractall(data_root)
tar.close()

```
**zipfile — Work with ZIP archives**


```
zipfile.ZipFile(file[, mode[, compression[, allowZip64]]])
```
Open a ZIP file, where file can be either a path to a file (a string) or a file-like object.
ZipFile is also a context manager and therefore supports the with statement.

Example:


```
with ZipFile('spam.zip', 'w') as myzip:
    myzip.write('eggs.txt')
```




```
ZipFile.read(name[, pwd])
```
Return the bytes of the file name in the archive. name is the name of the file in the archive, or a ZipInfo object. The archive must be open for read or append. pwd is the password used for encrypted files.



```
ZipFile.namelist()
```
Return a list of archive members by name.



### Data Types
** collections — High-performance container datatypes**


```
class collections.Counter([iterable-or-mapping])
```
A counter tool is provided to support convenient and rapid _tallies_. 

A Counter is a dict subclass for counting hashable objects. It is an unordered collection where elements are stored as dictionary keys and their counts are stored as dictionary values. 


```
>>> c = Counter()                           # a new, empty counter
>>> c = Counter('gallahad')                 # a new counter from an iterable
>>> c = Counter({'red': 4, 'blue': 2})      # a new counter from a mapping
>>> c = Counter(cats=4, dogs=8)             # a new counter from keyword args
```



```
most_common([n])
```

Return a list of the n most common elements and their counts from the most common to the least. If n is omitted or None, most_common() returns all elements in the counter. Elements with equal counts are ordered arbitrarily


```
>>> Counter('abracadabra').most_common(3)
[('a', 5), ('r', 2), ('b', 2)]
```
Each element is a tuple.

### Built-in Types
**
Comparisons**


```
Operation	Meaning
<	strictly less than
<=	less than or equal
>	strictly greater than
>=	greater than or equal
==	equal
!=	not equal
is	object identity
is not	negated object identity
```



**Numeric Types**


```
Operation	Result	
x + y	sum of x and y	 	 
x - y	difference of x and y	 	 
x * y	product of x and y	 	 
x / y	quotient of x and y	 	 
x // y	floored quotient of x and y	
x % y	remainder of x / y
pow(x, y)	x to the power y
x ** y	x to the power y

```


### Sequence Types — list, tuple, range



```
Operation	Result	
x in s	True if an item of s is equal to x, else False	
x not in s	False if an item of s is equal to x, else True	
s + t	the concatenation of s and t
s * n or n * s	equivalent to adding s to itself n times
s[i]	ith item of s, origin
s[i:j]	slice of s from i to j
s[i:j:k]	slice of s from i to j with step k
len(s)	length of s	 
min(s)	smallest item of s	 
max(s)	largest item of s	 
s.index(x[, i[, j]])	index of the first occurrence of x in s (at or after index i and before index j)
s.count(x)	total number of occurrences of x in s
```



**Lists**
Lists are mutable sequences, allow in-place modification of the object.
Using a pair of square brackets to denote the empty list: []
Using square brackets, separating items with commas: [a], [a, b, c]
Using a list comprehension: [x for x in iterable]


**Tuples**
Tuples are immutable sequences
Using a pair of parentheses to denote the empty tuple: ()
Using a trailing comma for a singleton tuple: a, or (a,)
Separating items with commas: a, b, c or (a, b, c)

**Ranges**
The range type represents an immutable sequence of numbers 
and is commonly used for looping a specific number of times in for loops.
class range(start, stop[, step])

**Mapping Types — dict**
A mapping object maps hashable values to arbitrary objects. Mappings are mutable objects.
Dictionaries can be created by placing a comma-separated list of key: value pairs within braces, 
for example: {'jack': 4098, 'sjoerd': 4127} or {4098: 'jack', 4127: 'sjoerd'}, or by the dict constructor.


```
>>> a = dict(one=1, two=2, three=3)
>>> b = {'one': 1, 'two': 2, 'three': 3}
>>> c = dict(zip(['one', 'two', 'three'], [1, 2, 3]))
>>> d = dict([('two', 2), ('one', 1), ('three', 3)])
>>> e = dict({'three': 3, 'one': 1, 'two': 2})
>>> a == b == c == d == e
True

```
keys()
Return a copy of the dictionary’s list of keys. See the note for dict.items().
values()
Return a copy of the dictionary’s list of values. See the note for dict.items().

### Built-in Functions


```
enumerate(iterable, start=0)
```


Return an enumerate object.
returns a tuple containing a count (from start which defaults to 0) and the values obtained from iterating over
iterable.



```
open(file, mode=’r’, buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None)
```


Open file and return a corresponding file object. If the file cannot be opened, an OSError is raised.

### Data Persistence
**pickle — Python object serialization**
“Pickling” is the process whereby a Python object hierarchy is converted into a byte stream, 
and “unpickling” is the inverse operation.



```
pickle.dump(obj, file[, protocol])
```


Write a pickled representation of obj to the open file object file. 


```
pickle.load(file[, *, encoding="ASCII", errors="strict"])
```


Read a pickled object representation from the open file object file 
and return the reconstituted object hierarchy specified therein.


```

>>>try:
>>>   with open(set_filename, 'wb') as f:
>>>       pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
>>>   except Exception as e:
>>>       print('Unable to save data to', set_filename, ':', e)
```



### Numeric and Mathematical Modules
**random — Generate pseudo-random numbers**


```
random.seed(a=None, version=2)
```


Initialize the random number generator.


```
random.choice(seq)
```


Return a random element from the non-empty sequence seq. If seq is empty, raises IndexError.


```
random.shuffle(x[, random])
```


Shuffle the sequence x in place.

### Cryptographic Services
**hashlib — Secure hashes and message digests**
For example: use sha256() to create a SHA-256 hash object. 
At any point you can ask it for the digest of the concatenation of the data fed to it so far 
using the digest() or hexdigest() methods.


```
>>> hashlib.sha224(b"Nobody inspects the spammish repetition").hexdigest()
```

### Simple statements
### assert
https://stackoverflow.com/questions/5142418/what-is-the-use-of-assert-in-python


```
assert condition
```
 you're telling the program to test that condition, and trigger an error if the condition is false.


```

if not condition:
    raise AssertionError()
```

