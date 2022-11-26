from typing import TypeAlias, Union, Tuple

import numpy as np

DirPath = str
FilePath = str
BGRColor = Tuple[int, int, int]

Array: TypeAlias = np.ndarray

BinaryFeature = Array['N', np.uint8]

# images
BGRImageArray = Array['H,W,3', np.uint8]
GrayImageArray = Array['H,W', np.uint8]

ImageArray = Union[BGRImageArray, GrayImageArray]

HeightPx = int
WidthPx = int
Channels = int