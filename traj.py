import dataclasses
from typing import Union


@dataclasses.dataclass(frozen=True)
class Vec2:
    x: float
    y: float


class Trajectory:
    def get_position(self, tt: float) -> Union[float, Vec2]:
        raise NotImplementedError

    def get_velocity(self, tt: float) -> Union[float, Vec2]:
        raise NotImplementedError

    def get_acceleration(self, tt: float) -> Union[float, Vec2]:
        raise NotImplementedError

    def get_total_time(self) -> Union[float, Vec2]:
        raise NotImplementedError
