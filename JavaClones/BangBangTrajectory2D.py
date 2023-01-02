import math
from typing import Callable

from JavaClones.BangBangTrajectory1D import BangBangTrajectory1D
from JavaClones.Trajectory import Trajectory, SYNC_ACCURACY
from JavaClones.Vec2 import Vec2


class BangBangTrajectory2D(Trajectory):
    x = BangBangTrajectory1D()
    y = BangBangTrajectory1D()
    alpha: float

    def get_position(self, t: float) -> Vec2:
        return Vec2(self.x.get_position(t), self.y.get_position(t))

    def get_velocity(self, t: float) -> Vec2:
        return Vec2(self.x.get_velocity(t), self.y.get_velocity(t))

    def get_acceleration(self, t: float) -> Vec2:
        return Vec2(self.x.get_acceleration(t), self.y.get_acceleration(t))

    def get_total_time(self) -> float:
        return max(self.x.get_total_time(), self.y.get_total_time())

    def generate(
            self,
            s0: Vec2,
            s1: Vec2,
            v0: Vec2,
            v_max: float,
            a_max: float,
            alpha_fn: Callable[[float], float]
    ) -> "BangBangTrajectory2D":

        s0x = s0.x
        s0y = s0.y
        s1x = s1.x
        s1y = s1.y
        v0x = v0.x
        v0y = v0.y

        inc = math.pi / 8.0
        alpha = math.pi / 4.0

        # binary search, some iterations (fixed)
        while inc > 1e-7:
            used_alpha = alpha_fn(alpha)
            sin_alpha = math.sin(used_alpha)
            cos_alpha = math.cos(used_alpha)

            self.x.generate(s0x, s1x, v0x, v_max * cos_alpha, a_max * cos_alpha)
            self.y.generate(s0y, s1y, v0y, v_max * sin_alpha, a_max * sin_alpha)

            diff = abs(self.x.get_total_time() - self.y.get_total_time())
            if diff < SYNC_ACCURACY:
                break
            if self.x.get_total_time() > self.y.get_total_time():
                alpha -= inc
            else:
                alpha += inc

            inc *= 0.5
        self.alpha = alpha
        return self
