from typing import TypeVar, Generic, TypeAlias, Union, Type

import numpy as np
from numpy.typing import ArrayLike

DirPath = str
FilePath = str


Shape = TypeVar("Shape")
DType = TypeVar("DType")


# class _Array(np.ndarray, Generic[Shape, DType]):
#     """ Generic type used in type annotations. For example,
#     x: Array['N,K,2', np.int]
#     is an np ndarray of `np.int`s of shape N, K, 2
#     """
#     pass
#


# Array = Union[_Array, np.ndarray]

# class Array(Generic[Shape, DType], Union[_Array[Shape, DType], np.ndarray]):
#     pass


# class Array(ArrayLike, Generic[Shape, DType]):
#     pass

# class Array(Generic[Shape, DType], ArrayLike):
#     pass

# class Array(Generic[Shape, DType], np.ndarray):
#     pass


Array = np.ndarray

"""

What do we want ???

- if we have function that wants Array['12', np.int], passing in np.ndarray is OK
- supports the generic syntax



"""