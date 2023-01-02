import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Vec2:
    x: float
    y: float

    @staticmethod
    def zero() -> "Vec2":
        return Vec2(0, 0)

    def __add__(self, other):
        if isinstance(other, (int, float)):
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
        if isinstance(other, (int, float)):
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
        return self.get_length2() < other.get_length2()

    def get_length(self) -> float:
        return math.sqrt(self.get_length2())

    def get_length2(self) -> float:
        return self.x ** 2 + self.y ** 2

    def get_angle(self) -> float:
        return math.atan2(self.y, self.x)

    def turn(self, angle: float) -> "Vec2":
        return Vec2(
            self.x * math.cos(angle) - self.y * math.sin(angle),
            self.y * math.cos(angle) + self.x * math.sin(angle)
        )

    def __str__(self):
        return "{:.3f}, {:.3f}".format(self.x, self.y)
