# Python and HDF5 

_----Hierarchical Data Format version 5_

**Keyword: structured, "self-describing"**
HDF5 is a great mechanism for storing large numerical arrays of homogenous type, and is just about perfect if you make minimal use of relational featues.
1. A file specification and associated data model.
2. A standard library with API
3. A software ecosystem, consisting of both client programs using HDF5 and "analysis platforms" like MATLAB and python.

2 "Killer features" of HDF5:
Organization in hierarchical groups and attributes.
Attributes let you attach descriptive metadata directly to the data it describes.

HDF5 Strengths:
Subsetting and partial I/O: the actual data lives on disk, the appropriate data is found and loaded into memory.
Control over how storage is allocated.

http://docs.h5py.org/en/latest/quick.html#core-concepts
## Core concepts
An HDF5 file is a container for two kinds of objects: **datasets**, which are array-like collections of data, and **groups**, which are folder-like containers that hold datasets and other groups. The most fundamental thing to remember when using h5py is:

_Groups work like dictionaries, and datasets work like NumPy arrays_

## File Objects
File objects serve as your entry point into the world of HDF5. 
### Opening & creating files
HDF5 files work generally like standard Python file objects. They support standard modes like r/w/a, and should be closed when they are no longer in use. However, there is obviously no concept of “text” vs “binary” mode.



```
>>> f = h5py.File('myfile.hdf5','r')
```



```
class File(name, mode=None, driver=None, libver=None, userblock_size, **kwds)
```

Open or create a new file.

Note that in addition to the File-specific methods and properties listed below, File objects inherit the full interface of Group.


### Documentation


```
Type:        Reference
String form: <HDF5 object reference>
Docstring:  
Opaque representation of an HDF5 reference.
```



Objects of this class are created exclusively by the library and
cannot be modified.  The read-only attribute "typecode" determines
whether the reference is to an object in an HDF5 file (OBJECT) or a dataset region (DATASET_REGION).

The object's truth value indicates whether it contains a nonzero
reference.  This does not guarantee that is valid, but is useful
for rejecting "background" elements in a dataset



### H5PY in python to get  data info from .mat file.

**Example: digitStruct.mat for street view house number**



```
Data Hierarchy 
[HDF5 Group: Parent Group]
-- #refs#
  [Sub Group ]
  -- Group with Ref No.****
      [HDF5 Scalar Dataset]
      --height
         No.of Dimension(s):2
         size: 3*1
         Data Type: Object reference
      --label
         No.of Dimension(s):2
         size: m*1 m<=5 at mot 5 digits
         Data Type: Object reference
      --left
         No.of Dimension(s):2
         size: 3*1
         Data Type: Object reference
      --top
         No.of Dimension(s):2
         size: 3*1
         Data Type: Object reference
      --width
         No.of Dimension(s):2
         size: 3*1
         Data Type: Object reference
   -- Dataset with Ref No.****
         No.of Dimension(s):2
         size: 1*1
         Data Type: 64-BIT floating-point

--digitStruct 
  [HDF5 Scalar Dataset]
   --bbox 
      No.of Dimension(s): 2
      size: 33402*1
      Data Type: Object reference
   --name
```




Steps:
1. Get the reference object in bbox.
digit_struct_mat_file['digitStruct']['bbox'][0].item()
2. Use the reference object  to get height, label, left, top and width
label = digit_struct_mat_file[item]['label']
3. Use the reference object to get the actual value.
digit_struct_mat_file[label[1].item()][0][0]

Notes:
-use name + '?' to get documentation in IPython.
-bang 'tab' after '.' to get hint.