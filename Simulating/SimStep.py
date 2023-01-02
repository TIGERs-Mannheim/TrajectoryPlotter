from dataclasses import dataclass
from typing import List, Union, Tuple, Optional

from JavaClones.BangBangTrajectory1D import BangBangTrajectory1D
from JavaClones.BangBangTrajectory2D import BangBangTrajectory2D
from JavaClones.BangBangTrajectory2DAsync import BangBangTrajectory2DAsync
from JavaClones.Vec2 import Vec2
from Simulating.AlphaData import AlphaData


@dataclass(frozen=True)
class SimStep:
    trajectory: Union[BangBangTrajectory1D, BangBangTrajectory2D, BangBangTrajectory2DAsync]
    step: int
    times: List[float]
    pos: Union[List[float], List[Vec2]]
    vel: Union[List[float], List[Vec2]]
    acc: Union[List[float], List[Vec2]]
    v_max: Union[float, Vec2]
    a_max: Union[float, Vec2]

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


@dataclass(frozen=True)
class SimStep1d(SimStep):
    trajectory: BangBangTrajectory1D
    pos: List[float]
    vel: List[float]
    acc: List[float]
    v_max: float
    a_max: float

    def split(self) -> Tuple[None, None]:
        return None, None


@dataclass(frozen=True)
class SimStep2d(SimStep):
    trajectory: Union[BangBangTrajectory2D, BangBangTrajectory2DAsync]
    pos: List[Vec2]
    vel: List[Vec2]
    acc: List[Vec2]
    alpha: float
    alpha_data: AlphaData
    v_max: Vec2
    a_max: Vec2

    def get_1d_x(self) -> SimStep1d:
        return self._get_1d("x")

    def get_1d_y(self) -> SimStep1d:
        return self._get_1d("y")

    def _get_1d(self, attr: str) -> SimStep1d:
        return SimStep1d(
            trajectory=getattr(self.trajectory, attr),
            step=self.step,
            times=self.times,
            pos=[getattr(i, attr) for i in self.pos],
            vel=[getattr(i, attr) for i in self.vel],
            acc=[getattr(i, attr) for i in self.acc],
            v_max=getattr(self.v_max, attr),
            a_max=getattr(self.a_max, attr),
        )

    def split(self) -> Tuple[SimStep1d, SimStep1d]:
        return self.get_1d_x(), self.get_1d_y()
