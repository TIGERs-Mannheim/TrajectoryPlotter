import dataclasses
import math
from typing import Tuple

from traj1D import BangBangTrajectory1D


@dataclasses.dataclass
class Vec2:
    x: float
    y: float


class BangBangTrajectory2D:
    x: BangBangTrajectory1D
    y: BangBangTrajectory1D

    def __int__(self, accuracy: float = None):
        self.x = BangBangTrajectory1D()
        self.y = BangBangTrajectory1D()

    def generate(self, initial_pos: Vec2, final_pos: Vec2, initial_vel: Vec2, max_vel: float, max_acc: float,
                 accuracy: float, target_time: float = None):
        if target_time is None or target_time < 0:
            self.generate_shortest(initial_pos, final_pos, initial_vel, max_vel, max_acc, accuracy)
        else:
            self.generate_timed(initial_pos, final_pos, initial_vel, max_vel, max_acc, accuracy, target_time)

    def generate_shortest(self, initial_pos: Vec2, final_pos: Vec2, initial_vel: Vec2, max_vel: float, max_acc: float,
                          accuracy: float):
        s0x = initial_pos.x
        s0y = initial_pos.y
        s1x = final_pos.x
        s1y = final_pos.y
        v0x = initial_vel.x
        v0y = initial_vel.y

        inc: float = math.pi / 8.0
        alpha: float = math.pi / 4.0

        # binary search, some iterations (fixed)
        while inc > 1e-7:
            sin_alpha = math.sin(alpha)
            cos_alpha = math.cos(alpha)

            self.x.generate(s0x, s1x, v0x, max_vel * cos_alpha, max_acc * cos_alpha)
            self.y.generate(s0y, s1y, v0y, max_vel * sin_alpha, max_acc * sin_alpha)

            diff = math.fabs(self.x.get_total_time() - self.y.get_total_time())
            if diff < accuracy:
                break

            if self.x.get_total_time() > self.y.get_total_time():

                alpha -= inc
            else:

                alpha += inc

            inc *= 0.5

    @staticmethod
    def split_vel_and_acc(alpha: float, max_vel: float, max_acc: float) -> Tuple[float, float, float, float]:
        sin_alpha = math.sin(alpha)
        cos_alpha = math.cos(alpha)

        max_vel_x = max_vel * cos_alpha
        max_vel_y = max_vel * sin_alpha

        max_acc_x = max_acc * cos_alpha
        max_acc_y = max_acc * sin_alpha

        return max_vel_x, max_vel_y, max_acc_x, max_acc_y

    def generate_timed(self, initial_pos: Vec2, final_pos: Vec2, initial_vel: Vec2, max_vel: float, max_acc: float,
                       accuracy: float, target_time: float = None):
        s0x = initial_pos.x
        s0y = initial_pos.y
        s1x = final_pos.x
        s1y = final_pos.y
        v0x = initial_vel.x
        v0y = initial_vel.y

        inc: float = math.pi / 8.0
        alpha: float = math.pi / 4.0

        # binary search, some iterations (fixed)
        while inc > 1e-7:
            max_vel_x, max_vel_y, max_acc_x, max_acc_y = self.split_vel_and_acc(alpha, max_vel, max_acc)

            _, _, _, time_remaining_x = BangBangTrajectory1D.can_reach(s0x, s1x, v0x, max_vel_x, max_acc_x, target_time)
            _, _, _, time_remaining_y = BangBangTrajectory1D.can_reach(s0y, s1y, v0y, max_vel_y, max_acc_y, target_time)

            diff = math.fabs(time_remaining_x - time_remaining_y)
            if diff < accuracy:
                break

            if time_remaining_x < time_remaining_y:

                alpha -= inc
            else:

                alpha += inc

            inc *= 0.5

        max_vel_x, max_vel_y, max_acc_x, max_acc_y = self.split_vel_and_acc(alpha, max_vel, max_acc)
        self.x.generate(s0x, s1x, v0x, max_vel_x, max_acc_x, target_time)
        self.y.generate(s0y, s1y, v0y, max_vel_y, max_acc_y, target_time)
