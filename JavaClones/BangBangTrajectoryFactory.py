from JavaClones.BangBangTrajectory1D import BangBangTrajectory1D
from JavaClones.BangBangTrajectory2D import BangBangTrajectory2D
from JavaClones.BangBangTrajectory2DAsync import BangBangTrajectory2DAsync
from JavaClones.Trajectory import alpha_fn_async
from JavaClones.Vec2 import Vec2


class BangBangTrajectoryFactory:

    @staticmethod
    def traj_2d_async(
            s0: Vec2,
            s1: Vec2,
            v0: Vec2,
            v_max: float,
            a_max: float,
            primary_direction: Vec2
    ) -> BangBangTrajectory2DAsync:
        rotation = primary_direction.get_angle()
        start_to_target = (s1 - s0).turn(-rotation)
        v0_rotated = v0.turn(-rotation)

        child = BangBangTrajectory2D().generate(
            Vec2.zero(),
            start_to_target,
            v0_rotated,
            v_max,
            a_max,
            alpha_fn_async
        )
        return BangBangTrajectory2DAsync(child, s0, rotation)

    @staticmethod
    def traj_2d_sync(
            s0: Vec2,
            s1: Vec2,
            v0: Vec2,
            v_max: float,
            a_max: float
    ) -> BangBangTrajectory2D:
        return BangBangTrajectory2D().generate(
            s0,
            s1,
            v0,
            v_max,
            a_max,
            lambda alpha: alpha
        )

    @staticmethod
    def traj_1d(
            s0: float,
            s1: float,
            v0: float,
            v_max: float,
            a_max: float
    ) -> BangBangTrajectory1D:
        return BangBangTrajectory1D().generate(
            s0,
            s1,
            v0,
            v_max,
            a_max
        )
