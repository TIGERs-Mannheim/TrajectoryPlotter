import math
from dataclasses import dataclass
from typing import List

from JavaClones.Trajectory import Trajectory

MAX_PARTS = 3


@dataclass
class BBTrajectoryPart:
    t_end: float = 0.0
    acc: float = 0.0
    v0: float = 0.0
    s0: float = 0.0


class BangBangTrajectory1D(Trajectory):
    parts: List[BBTrajectoryPart] = []
    numParts: int
    v_max: float
    a_max: float

    def __init__(self):
        self.parts = []
        for _ in range(MAX_PARTS):
            self.parts.append(BBTrajectoryPart())

    def get_position(self, tt: float) -> float:
        traj_time = max(0.0, tt)

        if traj_time >= self.get_total_time():
            # requested time beyond final element
            last_part = self.parts[self.numParts - 1]
            t = last_part.t_end - self.parts[self.numParts - 2].t_end
            return last_part.s0 + (last_part.v0 * t) + (0.5 * last_part.acc * t * t)

        piece_idx = self.find_part_idx(traj_time)
        piece = self.parts[piece_idx]
        t_piece_start = 0 if piece_idx < 1 else self.parts[piece_idx - 1].t_end
        t = traj_time - t_piece_start
        return piece.s0 + (piece.v0 * t) + (0.5 * piece.acc * t * t)

    def get_velocity(self, tt: float) -> float:
        traj_time = max(0.0, tt)

        if traj_time >= self.get_total_time():
            # requested time beyond final element
            return 0.0

        piece_idx = self.find_part_idx(traj_time)
        piece = self.parts[piece_idx]
        t_piece_start = 0 if piece_idx < 1 else self.parts[piece_idx - 1].t_end
        t = traj_time - t_piece_start
        return piece.v0 + (piece.acc * t)

    def get_acceleration(self, tt: float) -> float:
        traj_time = max(0.0, tt)

        if traj_time >= self.get_total_time():
            # requested time beyond final element
            return 0.0

        return self.find_part(traj_time).acc

    def get_total_time(self) -> float:
        return self.parts[self.numParts - 1].t_end

    def find_part_idx(self, t: float) -> int:
        for i in range(self.numParts):
            if t < self.parts[i].t_end:
                return i
        return self.numParts - 1

    def find_part(self, t: float) -> BBTrajectoryPart:
        return self.parts[self.find_part_idx(t)]

    def generate(
            self,
            initial_pos: float,
            final_pos: float,
            initial_vel: float,
            max_vel: float,
            max_acc: float
    ) -> "BangBangTrajectory1D":
        self.v_max = max_vel
        self.a_max = max_acc
        x0 = initial_pos
        xd0 = initial_vel
        x_trg = final_pos
        xd_max = max_vel
        xdd_max = max_acc
        s_at_zero_acc = self.vel_change_to_zero(x0, xd0, xdd_max)

        if s_at_zero_acc <= x_trg:
            s_end = self.vel_tri_to_zero(x0, xd0, xd_max, xdd_max)

            if s_end >= x_trg:
                # Triangular profile
                self.calc_tri(x0, xd0, x_trg, xdd_max)
            else:
                # Trapezoidal profile
                self.calc_trapz(x0, xd0, xd_max, x_trg, xdd_max)
        else:
            # even with a full brake we miss x_trg
            s_end = self.vel_tri_to_zero(x0, xd0, -xd_max, xdd_max)

            if s_end <= x_trg:
                # Triangular profile
                self.calc_tri(x0, xd0, x_trg, -xdd_max)
            else:
                # Trapezoidal profile
                self.calc_trapz(x0, xd0, -xd_max, x_trg, xdd_max)
        return self

    @staticmethod
    def vel_change_to_zero(s0: float, v0: float, a_max: float) -> float:
        if 0 >= v0:
            a = a_max
        else:
            a = -a_max

        t = -v0 / a
        return s0 + (0.5 * v0 * t)

    @staticmethod
    def vel_tri_to_zero(s0: float, v0: float, v1: float, a_max: float) -> float:
        if v1 >= v0:
            a1 = a_max
            a2 = -a_max
        else:
            a1 = -a_max
            a2 = a_max

        t1 = (v1 - v0) / a1
        s1 = s0 + (0.5 * (v0 + v1) * t1)

        t2 = -v1 / a2
        return s1 + (0.5 * v1 * t2)

    def calc_tri(
            self,
            s0: float,
            v0: float,
            s2: float,
            a: float
    ):
        t2: float
        v1: float
        t1: float
        s1: float
        sq: float

        if a > 0:
            # + -
            sq = ((a * (s2 - s0)) + (0.5 * v0 * v0)) / (a * a)
        else:
            # - +
            sq = ((-a * (s0 - s2)) + (0.5 * v0 * v0)) / (a * a)

        if sq > 0.0:
            t2 = math.sqrt(sq)
        else:
            t2 = 0

        v1 = a * t2
        t1 = (v1 - v0) / a
        s1 = s0 + ((v0 + v1) * 0.5 * t1)

        self.parts[0].t_end = t1
        self.parts[0].acc = a
        self.parts[0].v0 = v0
        self.parts[0].s0 = s0
        self.parts[1].t_end = t1 + t2
        self.parts[1].acc = -a
        self.parts[1].v0 = v1
        self.parts[1].s0 = s1
        self.numParts = 2

    def calc_trapz(
            self,
            s0: float,
            v0: float,
            v1: float,
            s3: float,
            a_max: float
    ):
        a1: float
        a3: float
        t1: float
        t2: float
        t3: float
        v2: float
        s1: float
        s2: float

        if v0 > v1:
            a1 = -a_max
        else:
            a1 = a_max

        if v1 > 0:
            a3 = -a_max
        else:
            a3 = a_max

        t1 = (v1 - v0) / a1
        v2 = v1
        t3 = -v2 / a3

        s1 = s0 + (0.5 * (v0 + v1) * t1)
        s2 = s3 - (0.5 * v2 * t3)
        t2 = (s2 - s1) / v1

        self.parts[0].t_end = t1
        self.parts[0].acc = a1
        self.parts[0].v0 = v0
        self.parts[0].s0 = s0
        self.parts[1].t_end = t1 + t2
        self.parts[1].acc = 0
        self.parts[1].v0 = v1
        self.parts[1].s0 = s1
        self.parts[2].t_end = t1 + t2 + t3
        self.parts[2].acc = a3
        self.parts[2].v0 = v2
        self.parts[2].s0 = s2
        self.numParts = 3
