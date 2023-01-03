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

    def __init__(self):
        self._config: SimConfig
        self._factory: BangBangTrajectoryFactory = BangBangTrajectoryFactory()
        self._pos_shifter: DestinationForTimedPositionCalc = DestinationForTimedPositionCalc()

    def __getattr__(self, item):
        return getattr(self._config, item)

    @staticmethod
    def simulate(config: SimConfig, num_steps: int, step_size: int):
        sim = Simulator()
        sim._config = config
        num_points = num_steps * step_size
        total_traj, alpha = sim._next_traj(Vec2.zero(), sim.v0, sim.tt)
        total_times = np.linspace(0, total_traj.get_total_time(), num_points)
        sim_steps: List[SimStep] = [sim._build_sim_step(0, total_times, total_traj, alpha, tt=sim.tt)]

        for step in range(1, num_steps + 1):
            next_index = min(step_size, len(sim_steps[-1].times) - 1)
            s0 = sim_steps[-1].pos[next_index]
            v0 = sim_steps[-1].vel[next_index]
            if sim.tt is None:
                tt = None
            else:
                tt = sim.tt - sim_steps[-1].times[next_index]

            times = sim_steps[-1].times[next_index:].copy()
            next_traj, alpha = sim._next_traj(s0, v0, tt)
            sim_steps.append(sim._build_sim_step(step, times, next_traj, alpha, tt))
        return sim_steps

    def _next_traj(self, s0: Union[float, Vec2], v0: Union[float, Vec2], tt: Optional[float]) \
            -> Tuple[Trajectory, float]:
        assert type(s0) == type(v0)
        if isinstance(s0, (float, int)):
            s1 = self.s
            if tt is not None:
                s1 = s0 + self._pos_shifter.get_timed_pos_1d(self.s - s0, v0, self.v_max, self.a_max, tt).pos
            return self._factory.traj_1d(s0, s1, v0, self.v_max, self.a_max), 0
        elif isinstance(s0, Vec2):
            if self.primary_direction is None:
                if tt is not None:
                    s1, alpha = self._pos_shifter.destination_for_bang_bang_2d_sync(
                        s0, self.s, v0, self.v_max, self.a_max, tt)
                    return self._factory.traj_2d_sync(s0, s1, v0, self.v_max, self.a_max), alpha
                else:
                    traj = self._factory.traj_2d_sync(s0, self.s, v0, self.v_max, self.a_max)
                    return traj, traj.alpha
            else:
                if tt is not None:
                    s1, alpha = self._pos_shifter.destination_for_bang_bang_2d_async(
                        s0, self.s, v0, self.v_max, self.a_max, tt, self.primary_direction)
                    return self._factory.traj_2d_async(s0, s1, v0, self.v_max, self.a_max,
                                                       self.primary_direction), alpha
                else:
                    traj = self._factory.traj_2d_async(s0, self.s, v0, self.v_max, self.a_max,
                                                       self.primary_direction)
                    return traj, traj.alpha
        else:
            raise ValueError("Unexpected Distance type {}".format(type(s0)))

    def _build_sim_step(self, current_step: int, step_times: Union[List[float], np.ndarray], traj: Trajectory,
                        alpha: float, tt: Optional[float]) -> SimStep:
        offset = 1e-6 if current_step != 0 else 0  # Offset the time a bit to avoid errors due to floating precision
        if isinstance(traj, BangBangTrajectory1D):
            return SimStep1d(
                trajectory=traj,
                step=current_step,
                times=step_times,
                pos=[traj.get_position(t - step_times[0] + offset) for t in step_times],
                vel=[traj.get_velocity(t - step_times[0] + offset) for t in step_times],
                acc=[traj.get_acceleration(t - step_times[0] + offset) for t in step_times],
                v_max=traj.v_max,
                a_max=traj.a_max,
                tt=tt,
            )
        elif isinstance(traj, (BangBangTrajectory2D, BangBangTrajectory2DAsync)):
            alpha_data = self.create_alpha_data(self._config, s0=traj.get_position(0), v0=traj.get_velocity(0), tt=tt)
            return SimStep2d(
                trajectory=traj,
                step=current_step,
                times=step_times,
                pos=[traj.get_position(t - step_times[0] + offset) for t in step_times],
                vel=[traj.get_velocity(t - step_times[0] + offset) for t in step_times],
                acc=[traj.get_acceleration(t - step_times[0] + offset) for t in step_times],
                alpha=alpha,
                alpha_data=alpha_data,
                v_max=Vec2(traj.x.v_max, traj.y.v_max),
                a_max=Vec2(traj.x.a_max, traj.y.a_max),
                tt=tt,
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
