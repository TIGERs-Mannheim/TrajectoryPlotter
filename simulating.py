import dataclasses
import math
from typing import List, Tuple, Union, Optional

import numpy as np

from traj import Trajectory, Vec2
from traj1D import BangBangTrajectory1D
from traj2D import BangBangTrajectory2D


@dataclasses.dataclass(frozen=True)
class SimStep:
    step: int
    times: List[float]
    pos: Union[List[float], List[Vec2]]
    vel: Union[List[float], List[Vec2]]
    acc: Union[List[float], List[Vec2]]
    max_vel: Union[float, Vec2]
    max_acc: Union[float, Vec2]

    def current_time(self) -> Union[float, Vec2]:
        return self.times[0]

    def current_pos(self) -> Union[float, Vec2]:
        return self.pos[0]

    def current_vel(self) -> Union[float, Vec2]:
        return self.vel[0]

    def current_acc(self) -> Union[float, Vec2]:
        return self.acc[0]

    def split(self) -> Tuple[Optional["SimStep1d"], Optional["SimStep1d"]]:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class SimStep1d(SimStep):
    pos: List[float]
    vel: List[float]
    acc: List[float]
    max_vel: float
    max_acc: float

    def split(self) -> Tuple[None, None]:
        return None, None


@dataclasses.dataclass(frozen=True)
class SimStep2d(SimStep):
    pos: List[Vec2]
    vel: List[Vec2]
    acc: List[Vec2]
    alpha: float
    optimal_alpha: float
    max_vel: Vec2
    max_acc: Vec2

    def get_1d_x(self) -> SimStep1d:
        return self._get_1d("x")

    def get_1d_y(self) -> SimStep1d:
        return self._get_1d("y")

    def _get_1d(self, attr: str) -> SimStep1d:
        return SimStep1d(
            step=self.step,
            times=self.times,
            pos=[getattr(i, attr) for i in self.pos],
            vel=[getattr(i, attr) for i in self.vel],
            acc=[getattr(i, attr) for i in self.acc],
            max_vel=getattr(self.max_vel, attr),
            max_acc=getattr(self.max_acc, attr),
        )

    def split(self) -> Tuple[SimStep1d, SimStep1d]:
        return self.get_1d_x(), self.get_1d_y()


@dataclasses.dataclass(frozen=True)
class Simulator:
    max_vel: float
    max_acc: float
    num_steps: int
    step_size: int
    distance: Union[float, Vec2]
    initial_vel: Union[float, Vec2]
    target_time: float = None

    def simulate(self) -> List[SimStep]:

        assert isinstance(self.distance, type(self.initial_vel))

        def traj(new_distance: Union[float, Vec2], new_initial_vel: Union[float, Vec2], new_target_time: float) \
                -> Trajectory:
            # print("({}), ({}), {}".format(new_distance, new_initial_vel, new_target_time))
            if isinstance(new_distance, (float, int)) and isinstance(new_initial_vel, float):
                new_traj = BangBangTrajectory1D()
                new_traj.generate(0, new_distance, new_initial_vel, self.max_vel, self.max_acc, new_target_time)
            elif isinstance(new_distance, Vec2) and isinstance(new_initial_vel, Vec2):
                new_traj = BangBangTrajectory2D()
                new_traj.generate(Vec2(0, 0), new_distance, new_initial_vel, self.max_vel, self.max_acc, 1e-3,
                                  new_target_time)
            else:
                raise ValueError("Unexpected Distance type {}".format(type(new_distance)))
            return new_traj

        def build_sim_step(current_step: int, step_times: Union[List[float], np.ndarray],
                           trajectory: Trajectory, pos_offset: Union[float, Vec2]) -> SimStep:
            if isinstance(trajectory, BangBangTrajectory1D):
                return SimStep1d(
                    step=current_step,
                    times=step_times,
                    pos=[pos_offset + trajectory.get_position(t - step_times[0]) for t in step_times],
                    vel=[trajectory.get_velocity(t - step_times[0]) for t in step_times],
                    acc=[trajectory.get_acceleration(t - step_times[0]) for t in step_times],
                    max_vel=trajectory.max_vel,
                    max_acc=trajectory.max_acc,
                )
            elif isinstance(trajectory, BangBangTrajectory2D):
                return SimStep2d(
                    step=current_step,
                    times=step_times,
                    pos=[pos_offset + trajectory.get_position(t - step_times[0]) for t in step_times],
                    vel=[trajectory.get_velocity(t - step_times[0]) for t in step_times],
                    acc=[trajectory.get_acceleration(t - step_times[0]) for t in step_times],
                    alpha=trajectory.alpha,
                    optimal_alpha=Simulator.optimal_alpha(trajectory, self.max_vel, self.max_acc, self.target_time),
                    max_vel=Vec2(trajectory.x.max_vel, trajectory.y.max_vel),
                    max_acc=Vec2(trajectory.x.max_acc, trajectory.y.max_acc),
                )
            else:
                raise ValueError

        num_points = self.num_steps * self.step_size
        total_traj = traj(self.distance, self.initial_vel, self.target_time)
        total_times = np.linspace(0, total_traj.get_total_time(), num_points)
        sim_steps: List[SimStep] = [build_sim_step(0, total_times, total_traj, pos_offset=0)]

        for step in range(1, self.num_steps + 1):
            s0 = sim_steps[-1].pos[min(self.step_size, len(sim_steps[-1].pos) - 1)]
            v0 = sim_steps[-1].vel[min(self.step_size, len(sim_steps[-1].vel) - 1)]
            tt = self.target_time - sim_steps[-1].times[
                min(self.step_size, len(sim_steps[-1].times) - 1)] if self.target_time is not None else None

            times = sim_steps[-1].times[min(self.step_size, len(sim_steps[-1].times) - 1):].copy()
            next_traj = traj(self.distance - s0, v0, tt)
            sim_steps.append(build_sim_step(step, times, next_traj, s0))
        return sim_steps

    @staticmethod
    def optimal_alpha(trajectory: BangBangTrajectory2D, max_vel, max_acc, target_time) -> float:
        initial_pos = trajectory.get_position(0.0)
        final_pos = trajectory.get_position(float("inf"))
        initial_vel = trajectory.get_velocity(0.0)
        alphas = np.linspace(1e-4, math.pi / 2.0 - 1e-4, num=1000)
        t_diffs = [BangBangTrajectory2D.diff_for_alpha(a, initial_pos, final_pos, initial_vel, max_vel,
                                                       max_acc, target_time)[0] for a in alphas]
        return alphas[np.argmin(t_diffs)]
