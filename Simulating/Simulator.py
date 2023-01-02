import math
from typing import Union, List, Optional, Tuple

import numpy as np

from JavaClones.BangBangTrajectory1D import BangBangTrajectory1D
from JavaClones.BangBangTrajectory2D import BangBangTrajectory2D
from JavaClones.BangBangTrajectory2DAsync import BangBangTrajectory2DAsync
from JavaClones.BangBangTrajectoryFactory import BangBangTrajectoryFactory, alpha_fn_async
from JavaClones.DestinationForTimedPositionCalc import DestinationForTimedPositionCalc
from JavaClones.Trajectory import Trajectory
from JavaClones.Vec2 import Vec2
from Simulating.AlphaData import AlphaData
from Simulating.SimConfig import SimConfig
from Simulating.SimStep import SimStep, SimStep1d, SimStep2d


class Simulator:
    _config: SimConfig
    _factory: BangBangTrajectoryFactory = BangBangTrajectoryFactory()
    _pos_shifter: DestinationForTimedPositionCalc = DestinationForTimedPositionCalc()

    def __getattr__(self, item):
        return getattr(self._config, item)

    def simulate(self, config: SimConfig, num_steps: int, step_size: int):
        self._config = config
        num_points = num_steps * step_size
        total_traj, alpha = self._next_traj(self.s, self.v0, self.tt)
        total_times = np.linspace(0, total_traj.get_total_time(), num_points)
        sim_steps: List[SimStep] = [self._build_sim_step(0, total_times, total_traj, alpha, pos_offset=0, tt=self.tt)]

        for step in range(1, num_steps + 1):
            s0 = sim_steps[-1].pos[min(step_size, len(sim_steps[-1].pos) - 1)]
            v0 = sim_steps[-1].vel[min(step_size, len(sim_steps[-1].vel) - 1)]
            if self.tt is None:
                tt = None
            else:
                tt = self.tt - sim_steps[-1].times[min(step_size, len(sim_steps[-1].times) - 1)]

            times = sim_steps[-1].times[min(step_size, len(sim_steps[-1].times) - 1):].copy()
            next_traj, alpha = self._next_traj(self.s - s0, v0, tt)
            sim_steps.append(self._build_sim_step(step, times, next_traj, alpha, s0, tt))
        return sim_steps

    def _next_traj(self, s: Union[float, Vec2], v0: Union[float, Vec2], tt: Optional[float]) \
            -> Tuple[Trajectory, float]:
        if isinstance(s, (float, int)):
            if tt is not None:
                s = self._pos_shifter.get_timed_pos_1d(s, v0, self.v_max, self.a_max, tt).pos
            return self._factory.traj_1d(0, s, v0, self.v_max, self.a_max), 0
        elif isinstance(s, Vec2):
            if self.primary_direction is None:
                if tt is not None:
                    s, alpha = self._pos_shifter.destination_for_bang_bang_2d_sync(
                        Vec2(0, 0), s, v0, self.v_max, self.a_max, tt)
                    return self._factory.traj_2d_sync(Vec2(0, 0), s, v0, self.v_max, self.a_max), alpha
                else:
                    traj = self._factory.traj_2d_sync(Vec2(0, 0), s, v0, self.v_max, self.a_max)
                    return traj, traj.alpha
            else:
                if tt is not None:
                    s, alpha = self._pos_shifter.destination_for_bang_bang_2d_async(
                        Vec2(0, 0), s, v0, self.v_max, self.a_max, tt, self.primary_direction)
                    return self._factory.traj_2d_async(Vec2(0, 0), s, v0, self.v_max, self.a_max,
                                                       self.primary_direction), alpha
                else:
                    traj = self._factory.traj_2d_async(Vec2(0, 0), s, v0, self.v_max, self.a_max,
                                                       self.primary_direction)
                    return traj, traj.alpha
        else:
            raise ValueError("Unexpected Distance type {}".format(type(s)))

    def _build_sim_step(self, current_step: int, step_times: Union[List[float], np.ndarray], traj: Trajectory,
                        alpha: float, pos_offset: Union[float, Vec2], tt: Optional[float]) -> SimStep:
        if isinstance(traj, BangBangTrajectory1D):
            return SimStep1d(
                trajectory=traj,
                step=current_step,
                times=step_times,
                pos=[pos_offset + traj.get_position(t - step_times[0]) for t in step_times],
                vel=[traj.get_velocity(t - step_times[0]) for t in step_times],
                acc=[traj.get_acceleration(t - step_times[0]) for t in step_times],
                v_max=traj.v_max,
                a_max=traj.a_max,
            )
        elif isinstance(traj, (BangBangTrajectory2D, BangBangTrajectory2DAsync)):
            alpha_data = self.create_alpha_data(self._config, s0=traj.get_position(0), v0=traj.get_velocity(0), tt=tt)
            return SimStep2d(
                trajectory=traj,
                step=current_step,
                times=step_times,
                pos=[pos_offset + traj.get_position(t - step_times[0]) for t in step_times],
                vel=[traj.get_velocity(t - step_times[0]) for t in step_times],
                acc=[traj.get_acceleration(t - step_times[0]) for t in step_times],
                alpha=alpha,
                alpha_data=alpha_data,
                v_max=Vec2(traj.x.v_max, traj.y.v_max),
                a_max=Vec2(traj.x.a_max, traj.y.a_max),
            )
        else:
            raise ValueError

    @staticmethod
    def create_alpha_data(config: SimConfig, s0: Optional[Vec2] = None, s1: Optional[Vec2] = None,
                          v0: Optional[Vec2] = None, v_max: Optional[float] = None, a_max: Optional[float] = None,
                          tt: Optional[float] = None) -> AlphaData:
        s0 = s0 if s0 is not None else Vec2.zero()
        s1 = s1 if s1 is not None else config.s
        v0 = v0 if v0 is not None else config.v0
        v_max = v_max if v_max is not None else config.v_max
        a_max = a_max if a_max is not None else config.a_max
        tt = tt if tt is not None else config.tt
        if config.primary_direction is None:
            def alpha_fn(a: float) -> float:
                return a
        else:
            alpha_fn = alpha_fn_async

        if tt is None:
            def time_fn(s0a: float, s1a: float, v0a: float, v_max_a: float, a_max_a: float):
                return BangBangTrajectory1D().generate(s0a, s1a, v0a, v_max_a, a_max_a).get_total_time()
        else:
            def time_fn(s0a: float, s1a: float, v0a: float, v_max_a: float, a_max_a: float):
                return DestinationForTimedPositionCalc.get_timed_pos_1d(s1a - s0a, v0a, v_max_a, a_max_a, tt).time
        alphas = np.linspace(1e-4, math.pi / 2.0 - 1e-4, num=1000)
        x_times = []
        y_times = []
        diffs = []

        for alpha in alphas:
            used_alpha = alpha_fn(alpha)
            sin_alpha = math.sin(used_alpha)
            cos_alpha = math.cos(used_alpha)
            x = time_fn(s0.x, s1.x, v0.x, v_max * cos_alpha, a_max * cos_alpha)
            y = time_fn(s0.y, s1.y, v0.y, v_max * sin_alpha, a_max * sin_alpha)

            diffs.append(abs(x - y))
            x_times.append(x)
            y_times.append(y)

        return AlphaData(
            alphas=list(alphas),
            x_times=x_times,
            y_times=y_times,
            diffs=diffs,
            optimal=alphas[np.argmin(diffs)]
        )
