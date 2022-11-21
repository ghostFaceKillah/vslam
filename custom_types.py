from typing import TypeAlias

import numpy as np

DirPath = str
FilePath = str

Array: TypeAlias = np.ndarray

BinaryFeature = Array['N', np.uint8]
