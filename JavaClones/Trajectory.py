import math
from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from JavaClones.Vec2 import Vec2

SYNC_ACCURACY = 1e-3
MAX_VEL_TOLERANCE = 0.2


def alpha_fn_async(alpha: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return alpha + ((math.pi / 2 - alpha) * 0.5)


class Trajectory(ABC):
    @abstractmethod
    def get_position(self, tt: float) -> Union[float, Vec2]:
        raise NotImplementedError

    @abstractmethod
    def get_velocity(self, tt: float) -> Union[float, Vec2]:
        raise NotImplementedError

    @abstractmethod
    def get_acceleration(self, tt: float) -> Union[float, Vec2]:
        raise NotImplementedError

    @abstractmethod
    def get_total_time(self) -> float:
        raise NotImplementedError
