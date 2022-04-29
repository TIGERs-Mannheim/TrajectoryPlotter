import math
from typing import Tuple

from traj import Trajectory, Vec2
from traj1D import BangBangTrajectory1D


class BangBangTrajectory2D(Trajectory):
    alpha: float

    def get_position(self, tt: float) -> Vec2:
        return Vec2(self.x.get_position(tt), self.y.get_position(tt))

    def get_velocity(self, tt: float) -> Vec2:
        return Vec2(self.x.get_velocity(tt), self.y.get_velocity(tt))

    def get_acceleration(self, tt: float) -> Vec2:
        return Vec2(self.x.get_acceleration(tt), self.y.get_acceleration(tt))

    def get_total_time(self) -> float:
        return max(self.x.get_total_time(), self.y.get_total_time())

    x: BangBangTrajectory1D
    y: BangBangTrajectory1D

    def __init__(self):
        self.x = BangBangTrajectory1D()
        self.y = BangBangTrajectory1D()

    def generate(self, initial_pos: Vec2, final_pos: Vec2, initial_vel: Vec2, max_vel: float, max_acc: float,
                 accuracy: float, target_time: float):
        self.generate_slow(initial_pos, final_pos, initial_vel, max_vel, max_acc, accuracy, target_time)
        return
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
            self.alpha = alpha
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
                       accuracy: float, target_time: float):
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
            self.alpha = alpha
            max_vel_x, max_vel_y, max_acc_x, max_acc_y = self.split_vel_and_acc(alpha, max_vel, max_acc)

            _, _, _, time_remaining_x = BangBangTrajectory1D.can_reach(s0x, s1x, v0x, max_vel_x, max_acc_x, target_time)
            _, _, _, time_remaining_y = BangBangTrajectory1D.can_reach(s0y, s1y, v0y, max_vel_y, max_acc_y, target_time)

            diff = math.fabs(time_remaining_x - time_remaining_y)
            # print("{:.6f} - {:.6f}   |   {}".format(time_remaining_x, time_remaining_y, diff))
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

    def generate_slow(self, initial_pos: Vec2, final_pos: Vec2, initial_vel: Vec2, max_vel: float, max_acc: float,
                      accuracy: float, target_time: float = None):

        inc: float = math.pi / 8.0
        self.alpha: float = math.pi / 4.0

        while inc > 1e-7:
            l_alpha = self.alpha - inc
            r_alpha = self.alpha + inc

            l_diff, _, _ = self.diff_for_alpha(l_alpha, initial_pos, final_pos, initial_vel, max_vel, max_acc,
                                               target_time)
            r_diff, _, _ = self.diff_for_alpha(r_alpha, initial_pos, final_pos, initial_vel, max_vel, max_acc,
                                               target_time)

            left_is_better = l_diff <= r_diff
            if left_is_better:
                self.alpha = l_alpha
                if l_diff < accuracy:
                    break
            else:
                self.alpha = r_alpha
                if r_diff < accuracy:
                    break
            inc *= 0.5

        s0x = initial_pos.x
        s0y = initial_pos.y
        s1x = final_pos.x
        s1y = final_pos.y
        v0x = initial_vel.x
        v0y = initial_vel.y
        max_vel_x, max_vel_y, max_acc_x, max_acc_y = self.split_vel_and_acc(self.alpha, max_vel, max_acc)
        self.x.generate(s0x, s1x, v0x, max_vel_x, max_acc_x, target_time)
        self.y.generate(s0y, s1y, v0y, max_vel_y, max_acc_y, target_time)

    @staticmethod
    def diff_for_alpha(alpha: float, initial_pos: Vec2, final_pos: Vec2, initial_vel: Vec2, max_vel: float,
                       max_acc: float, target_time: float = None):
        s0x = initial_pos.x
        s0y = initial_pos.y
        s1x = final_pos.x
        s1y = final_pos.y
        v0x = initial_vel.x
        v0y = initial_vel.y

        max_vel_x, max_vel_y, max_acc_x, max_acc_y = BangBangTrajectory2D.split_vel_and_acc(alpha, max_vel, max_acc)

        if target_time is None or target_time <= 0:
            x_total = BangBangTrajectory1D().generate(s0x, s1x, v0x, max_vel_x, max_acc_x, None).get_total_time()
            y_total = BangBangTrajectory1D().generate(s0y, s1y, v0y, max_vel_y, max_acc_y, None).get_total_time()

            return math.fabs(x_total - y_total), x_total, y_total
        else:
            _, _, _, time_remaining_x = BangBangTrajectory1D.can_reach(s0x, s1x, v0x, max_vel_x, max_acc_x, target_time)
            _, _, _, time_remaining_y = BangBangTrajectory1D.can_reach(s0y, s1y, v0y, max_vel_y, max_acc_y, target_time)
            return math.fabs(time_remaining_x - time_remaining_y), -time_remaining_x, -time_remaining_y
