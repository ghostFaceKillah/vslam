from typing import TypeAlias, Union

import numpy as np

DirPath = str
FilePath = str

Array: TypeAlias = np.ndarray

BinaryFeature = Array['N', np.uint8]

# images
BGRImageArray = Array['H,W,3', np.uint8]
GrayImageArray = Array['H,W', np.uint8]

ImageArray = Union[BGRImageArray, GrayImageArray]