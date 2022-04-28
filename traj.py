import dataclasses
import math
from typing import Union


@dataclasses.dataclass(frozen=True)
class Vec2:
    x: float
    y: float

    def __add__(self, other):
        if isinstance(other, int):
            return Vec2(
                self.x + other,
                self.y + other,
            )
        elif isinstance(other, float):
            return Vec2(
                self.x + other,
                self.y + other,
            )
        elif isinstance(other, Vec2):
            return Vec2(
                other.x + self.x,
                other.y + self.y
            )
        raise NotImplementedError("{}".format(type(other)))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, int):
            return Vec2(
                self.x - other,
                self.y - other,
            )
        elif isinstance(other, float):
            return Vec2(
                self.x - other,
                self.y - other,
            )
        elif isinstance(other, Vec2):
            return Vec2(
                self.x - other.x,
                self.y - other.y
            )
        raise NotImplementedError("{}".format(type(other)))

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Vec2(
                other - self.x,
                other - self.y,
            )
        elif isinstance(other, Vec2):
            return Vec2(
                other.x - self.x,
                other.y - self.y
            )
        raise NotImplementedError("{}".format(type(other)))

    def __lt__(self, other):
        return self.length() < other.length()

    def length(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def __str__(self):
        return "{:.3f}, {:.3f}".format(self.x, self.y)


class Trajectory:
    def get_position(self, tt: float) -> Union[float, Vec2]:
        raise NotImplementedError

    def get_velocity(self, tt: float) -> Union[float, Vec2]:
        raise NotImplementedError

    def get_acceleration(self, tt: float) -> Union[float, Vec2]:
        raise NotImplementedError

    def get_total_time(self) -> Union[float, Vec2]:
        raise NotImplementedError
