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
MaskArray = Array['H,W', bool]

ImageArray = Union[BGRImageArray, GrayImageArray]

HeightPx = int
WidthPx = int
Channels = int

Pixel = Tuple[int, int]    # height, width, non-negative
PixelCoordArray = Array['N,2', np.int32]

OpenCVPixel = Tuple[int, int]   # width, height, non-negative
