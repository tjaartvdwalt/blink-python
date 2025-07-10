from dataclasses import dataclass
from enum import Enum

from cv2.typing import NumPyArrayFloat64, NumPyArrayNumeric
import numpy as np


class BlinkState(Enum):
    undefined = -1
    no = 0
    start = 1
    continued = 2


class EyeState(Enum):
    undefined = -1
    open = 0
    partial = 1
    closed = 2


