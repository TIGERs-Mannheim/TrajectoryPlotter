from JavaClones.BangBangTrajectory2D import BangBangTrajectory2D
from JavaClones.Trajectory import Trajectory
from JavaClones.Vec2 import Vec2


class BangBangTrajectory2DAsync(Trajectory):

    def __init__(self, child: BangBangTrajectory2D, initial_pos: Vec2, rotation: float):
        self.child = child
        self.initial_pos = initial_pos
        self.rotation = rotation

    def __getattr__(self, item):
        return getattr(self.child, item)

    def get_position(self, t: float) -> Vec2:
        return self.child.get_position(t).turn(self.rotation) + self.initial_pos

    def get_velocity(self, t: float) -> Vec2:
        return self.child.get_velocity(t).turn(self.rotation)

    def get_acceleration(self, t: float) -> Vec2:
        return self.child.get_acceleration(t).turn(self.rotation)

    def get_total_time(self) -> float:
        return self.child.get_total_time()

    def get_total_time_to_primary_direction(self):
        return self.child.y.get_total_time()
