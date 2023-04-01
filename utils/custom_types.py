from typing import TypeAlias, Union, Tuple

import jax.numpy as jnp
import numpy as np

DirPath = str
FilePath = str
BGRColor = Tuple[int, int, int]

Array: TypeAlias = np.ndarray
JaxArray: TypeAlias = jnp.ndarray   # as of writing, jax has only partial support for type annotation

BinaryFeature = Array['N', np.uint8]

# images
BGRImageArray = Array['H,W,3', np.uint8]
GrayImageArray = Array['H,W', np.uint8]
MaskArray = Array['H,W', bool]

ImageArray = Union[BGRImageArray, GrayImageArray]
JaxImageArray: TypeAlias = JaxArray  # ['H,W,3', jnp.uint8]  # that's sad, but I won't fight it just now :)

HeightPx = int
WidthPx = int
Channels = int

Pixel = Tuple[int, int]    # down, right, non-negative
PixelCoordArray = Array['N,2', np.int32]

OpenCVPixel = Tuple[int, int]   # right, down, non-negative
