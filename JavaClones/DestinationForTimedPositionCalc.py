import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from JavaClones.Trajectory import alpha_fn_async, SYNC_ACCURACY
from JavaClones.Vec2 import Vec2


@dataclass(frozen=True)
class TimedPos1D:
    pos: float
    time: float


class DestinationForTimedPositionCalc:

    @staticmethod
    def destination_for_bang_bang_2d_sync(
            s0: Vec2,
            s1: Vec2,
            v0: Vec2,
            v_max: float,
            a_max: float,
            tt: float
    ) -> Tuple[Vec2, float]:

        return DestinationForTimedPositionCalc.destination_for_bang_bang_2d(
            s0,
            s1,
            v0,
            v_max,
            a_max,
            tt,
            lambda alpha: alpha
        )

    @staticmethod
    def destination_for_bang_bang_2d_async(
            s0: Vec2,
            s1: Vec2,
            v0: Vec2,
            v_max: float,
            a_max: float,
            tt: float,
            primary_direction: Vec2
    ) -> Tuple[Vec2, float]:

        rotation = primary_direction.get_angle()
        start_to_target = (s1 - s0).turn(-rotation)
        v0_rotated = v0.turn(-rotation)

        res = DestinationForTimedPositionCalc.destination_for_bang_bang_2d(
            Vec2.zero(),
            start_to_target,
            v0_rotated,
            v_max,
            a_max,
            tt,
            alpha_fn_async
        )
        return res[0].turn(rotation) + s0, res[1]

    @staticmethod
    def destination_for_bang_bang_2d(
            s0: Vec2,
            s1: Vec2,
            v0: Vec2,
            v_max: float,
            a_max: float,
            tt: float,
            alpha_fn: Callable[[float], float]
    ) -> Tuple[Vec2, float]:

        v0x = v0.x
        v0y = v0.y
        distance = (s1 - s0)
        distance_x = distance.x
        distance_y = distance.y

        inc = math.pi / 8.0
        alpha = math.pi / 4.0

        x = TimedPos1D(0, 0)
        y = TimedPos1D(0, 0)

        # binary search, some iterations (fixed)
        while inc > 1e-7:
            used_alpha = alpha_fn(alpha)
            s_a = math.sin(used_alpha)
            c_a = math.cos(used_alpha)

            x = DestinationForTimedPositionCalc.get_timed_pos_1d(distance_x, v0x, v_max * c_a, a_max * c_a, tt)
            y = DestinationForTimedPositionCalc.get_timed_pos_1d(distance_y, v0y, v_max * s_a, a_max * s_a, tt)

            if abs(x.time - y.time) < SYNC_ACCURACY:
                break
            elif x.time > y.time:
                alpha -= inc
            else:
                alpha += inc

            inc *= 0.5

        return Vec2(x.pos + s0.x, y.pos + s0.y), alpha

    @staticmethod
    def get_timed_pos_1d(s: float, v0: float, v_max: float, a_max: float, tt: float) -> TimedPos1D:

        # Hit Windows:
        # Either our v0 is low enough that we could stop before reaching the goal target:
        #
        #  |    |----------------------------------------------------
        #  |             Direct Hit
        # -|---------------------------------------------------------> tt
        #
        # Or our v0 is so high that we will always overshoot. Is target time low enough to direct hit or do we
        # need to overshoot and recover?
        #
        #  |    |----------|                |------------------------
        #  |     Direct Hit                   Overshoot and Recover
        # -|---------------------------------------------------------> tt

        a_dec = -a_max if v0 >= 0 else a_max
        t_breaking = (-v0 / a_dec)
        s_zero_vel = 0.5 * v0 * t_breaking
        v1_max = v_max if s >= 0 else -v_max

        if ((s >= 0.0) != (v0 > 0.0)  # If v0 and (s0 -> sT) are in a different direction -> no forced overshoot
                # abs(s_zero_vel) > abs(s), without this breaking is always possible -> no forced overshoot
                or (s >= 0) == (s_zero_vel < s)
                # Determine if we can be slow enough -> no forced overshoot
                or DestinationForTimedPositionCalc.calc_slowest_direct_time(s, v0, a_max) >= tt):
            # We can directly hit the timed target position
            return DestinationForTimedPositionCalc.calc_fastest_direct(s, v0, v1_max, a_max, tt)
        else:
            # Calc the fastest overshoot by starting at s_zero_vel in opposed direction with v0=0.0
            timed = DestinationForTimedPositionCalc.calc_fastest_direct(s - s_zero_vel, 0.0, -v1_max, a_max,
                                                                        tt - t_breaking)
            # Extend TimedPos1D to accommodate breaking
            return TimedPos1D(timed.pos + s_zero_vel, timed.time + t_breaking)

    @staticmethod
    def calc_slowest_direct_time(s: float, v0: float, a_max: float) -> float:

        a_dec = -a_max if (v0 >= 0) else a_max
        sqrt = math.sqrt(v0 * v0 + 2 * a_dec * s)
        return ((-v0 + sqrt) / a_dec) if (v0 >= 0.0) else ((-v0 - sqrt) / a_dec)

    @staticmethod
    def calc_fastest_direct(
            s: float,
            v0: float,
            v1_max: float,
            a_max: float,
            tt: float
    ) -> TimedPos1D:

        # Possible Fastest Directs:
        #  - Straight too slow
        #  - Trapezoidal too slow
        #  - Trapezoidal finishing early
        #  - Trapezoidal direct hit
        #  - Triangular too slow
        #  - Triangular finishing early
        #  - Triangular direct hit
        a_dec = -a_max if v1_max >= 0 else a_max
        trapezoidal = DestinationForTimedPositionCalc.calc_fastest_direct_trapezoidal(s, v0, v1_max, a_max, a_dec, tt)
        if trapezoidal is not None:
            return trapezoidal
        else:
            return DestinationForTimedPositionCalc.calc_fastest_direct_triangular(s, v0, v1_max, a_max, a_dec, tt)

    @staticmethod
    def calc_fastest_direct_trapezoidal(
            s: float,
            v0: float,
            v1_max: float,
            a_max: float,
            a_dec: float,
            tt: float
    ) -> Optional[TimedPos1D]:

        # Full acceleration for s01 to reach v1_max
        a_acc = -a_max if v0 >= v1_max else a_max
        t01 = (v1_max - v0) / a_acc
        s01 = 0.5 * (v1_max + v0) * t01

        if (s >= 0.0) == (s <= s01):
            # We are not able to accel to v1_max before reaching s -> No Trapezoidal form possible
            return None

        s13 = s - s01
        t23 = -v1_max / a_dec
        s23 = 0.5 * v1_max * t23

        # Determining if "Trapezoidal too slow"
        # v1_max,v2   _________
        #            /|   |   |\
        #           / |   |   | \          s reached at t=t2
        #          /  |   |   |  \
        #         /   |   |   |   \
        # v0,v3  /    |   |   |    \
        #    ---|-----|---|---|-----|----------->
        #      t0    t1  tt  t2    t3
        #       |-t01-|--t12--|-t23-|
        t12_too_slow = s13 / v1_max
        if t01 + t12_too_slow >= tt:
            return TimedPos1D(s + s23, t01 + t12_too_slow + t23)

        # Determine if "Trapezoidal finishing early"
        # v1_max,v2   _________
        #            /|       |\
        #           / |       | \          s reached at t=t3
        #          /  |       |  \
        #         /   |       |   \
        # v0,v3  /    |       |    \
        #    ---|-----|-------|-----|-----|----->
        #      t0    t1      t2    t3    tt
        #       |-t01-|--t12--|-t23-|
        s12_early = s13 - s23
        t12_early = s12_early / v1_max
        if t12_early >= 0.0 and t01 + t12_early + t23 <= tt:
            return TimedPos1D(s, t01 + t12_early + t23)

        # Determine if "Trapezoidal direct hit"
        # v1_max,v2    _________
        #             /|       |\
        #            / |       | \         tt = t3
        #           /  |       |  \        s reached at t=tt
        # v3       /   |       |   \
        #         /    |       |   |\
        # v0,v4  /     |       |   | \
        #    ---|------|-------|---|--|--------->
        #      t0     t1      t2  tt t4
        #       |-t01--|--t12--|t23|
        #              |----t13----|
        # https://www.wolframalpha.com/input?i=solve+v_0*t_1+%3Dv_0*t_2%2B1%2F2*a*Power%5Bt_2%2C2%5D%2Bv_1*t_3%2C+v_1+%3D+v_0%2Ba*t_2%2C+t_1%2Bt%3Dt_2%2Bt_3+for+v_1%2Ct_1%2C++t_2
        t13 = tt - t01
        t23_direct = math.sqrt(2 * (s13 - t13 * v1_max) / a_dec)
        t12_direct = t13 - t23_direct
        if t12_direct > 0 and t23_direct < t23:
            v3 = v1_max + a_dec * t23_direct
            t34 = -v3 / a_dec
            return TimedPos1D(s + 0.5 * v3 * t34, tt + t34)

        return None

    @staticmethod
    def calc_fastest_direct_triangular(
            s: float,
            v0: float,
            v1_max: float,
            a_max: float,
            a_dec: float,
            tt: float
    ) -> TimedPos1D:

        # Determining if "Straight too slow"
        # Cant reach v1_max before reaching s, but already checked slowestDirect Time is smaller than tt (getPosition1D)
        # => we are too slow at s and only reasonable trajectory left is straight decelerating
        if (v1_max >= 0) == (v0 >= v1_max):
            t = -v0 / a_dec
            return TimedPos1D(0.5 * v0 * t, t)

        a_acc = -a_dec
        # Determining if "Triangular too slow"
        #
        #                  _/|\_
        #                _/  |  \_
        #              _/    |    \_
        #            _/|     |      \_        s reached at t=t1
        #          _/  |     |        \_
        #        _/    |     |          \_
        #    ---|------|-----|------------|-------->
        #      t0     tt    t1           t2
        #       |-----t01----|-----t12----|
        # https://www.wolframalpha.com/input?i=solve+s%3Dv_0*t_01%2B1%2F2*a*t_01%5E2%2C+for+t_01
        sqrt_too_slow = math.sqrt(2 * a_acc * s + v0 * v0)
        t01_too_s_low = ((-v0 + sqrt_too_slow) / a_acc) if (v1_max >= 0.0) else ((-v0 - sqrt_too_slow) / a_acc)
        if t01_too_s_low >= tt:
            v1_too_slow = v0 + a_acc * t01_too_s_low
            t12_too_slow = abs(v1_too_slow / a_acc)
            return TimedPos1D(s + 0.5 * v1_too_slow * t12_too_slow, t01_too_s_low + t12_too_slow)

        # Determining if "Triangular finishing early"
        #
        #                  _/|\_
        #                _/  |  \_
        #              _/    |    \_
        #            _/      |      \_        s reached at t=t2
        #          _/        |        \_
        #        _/          |          \_
        #    ---|------------|------------|----|--->
        #      t0           t1           t2   tt
        #       |-----t01----|-----t12----|
        # https://www.wolframalpha.com/input?i=solve+s%3Dv_0+*+t_1+%2B+1%2F2+*+a+*+t_1%5E2+%2B+v_1+*+t_2+%2B+1%2F2+*+%28-a%29+*+t_2%5E2%2C+v_1+%3D+v_0+%2B+a+*+t_1%2C+0%3Dv_1%2B+%28-a%29+*+t_2+for+t_1%2C+v_1%2C+t_2
        sq_early = ((s * a_acc) + (0.5 * v0 * v0)) / (a_max * a_max)
        t12_early = math.sqrt(sq_early) if sq_early > 0.0 else 0.0
        v1_early = a_acc * t12_early
        t01_early = (v1_early - v0) / a_acc
        if t01_early + t12_early <= tt:
            return TimedPos1D(s, t01_early + t12_early)

        # Determining if "Triangular direct hit"
        #
        #                  _/|\_
        #                _/  |  \_
        #              _/    |    \_
        #            _/      |     |\_        s reached at t=tt
        #          _/        |     |  \_
        #        _/          |     |    \_
        #    ---|------------|-----|------|-------->
        #      t0           t1    tt     t3
        #       |-----t01----|-t12-|-t23--|
        #                    |-----t13----|
        # https://www.wolframalpha.com/input?i=solve+s+%3D+v_0+*+t_1+%2B+0.5+*+a+*+t_1+**+2+%2Bv_1*t_2+-+0.5+*+a+*+t_2**2%2C+v_1+%3D+v_0+%2B+a+*+t_1%2C+v_2%3Dv_1-a*t_2%2C+t%3Dt_1%2Bt_2+for+v_2%2Cv_1%2C+t_1%2C+t_2

        sq_direct = math.sqrt(2 * a_acc * (a_acc * tt * tt - 2 * s + 2 * tt * v0))
        t01_direct = tt - sq_direct / (2 * a_max)
        v1_direct = v0 + a_acc * t01_direct
        t13_direct = v1_direct / a_acc
        s01_direct = 0.5 * (v0 + v1_direct) * t01_direct
        s13_direct = 0.5 * v1_direct * t13_direct
        return TimedPos1D(s01_direct + s13_direct, t01_direct + t13_direct)
