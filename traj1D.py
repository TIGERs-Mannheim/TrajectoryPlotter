import math
from typing import List, Optional

ACCURACY = 0.001
CONSTANT_SPEED_DURATION = 1.0


class BBTrajectoryPart:
    t_end: float = 0.0
    acc: float = 0.0
    v0: float = 0.0
    s0: float = 0.0


class BangBangTrajectory1D:
    def __init__(self):
        self.parts: List[BBTrajectoryPart] = []

    def getPosition(self, tt: float) -> float:
        traj_time: float = max(0.0, tt)

        if traj_time >= self.getTotalTime():
            # requested time beyond final element
            last_part = self.parts[len(self.parts) - 1]
            t = last_part.t_end - self.parts[len(self.parts) - 2].t_end
            return last_part.s0 + (last_part.v0 * t) + (0.5 * last_part.acc * t * t)

        piece_idx = self.findPartIdx(traj_time)
        piece = self.parts[piece_idx]
        t_piece_start = 0 if piece_idx < 1 else self.parts[piece_idx - 1].t_end
        t = traj_time - t_piece_start
        return piece.s0 + (piece.v0 * t) + (0.5 * piece.acc * t * t)

    def getVelocity(self, tt: float) -> float:
        traj_time = max(0.0, tt)

        if traj_time >= self.getTotalTime():
            # requested time beyond final element
            return 0.0

        piece_idx = self.findPartIdx(traj_time)
        piece = self.parts[piece_idx]
        t_piece_start = 0 if piece_idx < 1 else self.parts[piece_idx - 1].t_end
        t = traj_time - t_piece_start
        return piece.v0 + (piece.acc * t)

    def getAcceleration(self, tt: float) -> float:
        traj_time = max(0.0, tt)

        if traj_time >= self.getTotalTime():
            # requested time beyond final element
            return 0.0

        return self.findPart(traj_time).acc

    def getTotalTime(self) -> float:
        return self.parts[len(self.parts) - 1].t_end

    def findPartIdx(self, t: float) -> int:
        for i in range(len(self.parts)):
            if t < self.parts[i].t_end:
                return i
        return len(self.parts) - 1

    def findPart(self, t: float) -> BBTrajectoryPart:
        return self.parts[self.findPartIdx(t)]

    def generate(self, initial_pos: float, final_pos: float, initial_vel: float, max_vel: float, max_acc: float,
                 target_time=None):
        if target_time is None:
            self.checkAndAppendParts(
                BangBangTrajectory1D.generateShortest(initial_pos, final_pos, initial_vel, max_vel, max_acc))
        else:
            self.checkAndAppendParts(
                BangBangTrajectory1D.generateTimed(initial_pos, final_pos, initial_vel, max_vel, max_acc, target_time))

    def checkAndAppendParts(self, parts: List[BBTrajectoryPart]):
        self.parts = BangBangTrajectory1D.checkAndCombineParts(self.parts, parts)

    @staticmethod
    def generateShortest(initial_pos: float, final_pos: float, initial_vel: float, max_vel: float,
                         max_acc: float) -> List[BBTrajectoryPart]:

        x0 = initial_pos
        xd0 = initial_vel
        x_trg = final_pos
        xd_max = max_vel
        xdd_max = max_acc
        s_at_zero_acc = BangBangTrajectory1D.velChangeToZero(x0, xd0, xdd_max)

        if s_at_zero_acc <= x_trg:
            s_end = BangBangTrajectory1D.velTriToZero(x0, xd0, xd_max, xdd_max)
            if s_end >= x_trg:
                # Triangular profile
                return BangBangTrajectory1D.calcTri(x0, xd0, x_trg, xdd_max)
            else:
                # Trapezoidal profile
                return BangBangTrajectory1D.calcTrapz(x0, xd0, xd_max, x_trg, xdd_max)

        else:
            # even with a full brake we miss x_trg
            s_end = BangBangTrajectory1D.velTriToZero(x0, xd0, -xd_max, xdd_max)
            if s_end <= x_trg:
                # Triangular profile
                return BangBangTrajectory1D.calcTri(x0, xd0, x_trg, -xdd_max)
            else:
                # Trapezoidal profile
                return BangBangTrajectory1D.calcTrapz(x0, xd0, -xd_max, x_trg, xdd_max)

    @staticmethod
    def generateTimed(initial_pos: float, final_pos: float, initial_vel: float, max_vel: float, max_acc: float,
                      target_time) -> List[BBTrajectoryPart]:

        def finish_up(overshooting_parts: List[BBTrajectoryPart]):
            BangBangTrajectory1D.appendConstantSpeedPart(overshooting_parts)
            t_diff = overshooting_parts[-1].t_end - overshooting_parts[-2].t_end \
                if len(overshooting_parts) > 1 \
                else overshooting_parts[-1].t_end
            return BangBangTrajectory1D.checkAndCombineParts(overshooting_parts, BangBangTrajectory1D.generateShortest(
                initial_pos=overshooting_parts[-1].s0 + overshooting_parts[-1].v0 * t_diff,
                final_pos=final_pos,
                initial_vel=overshooting_parts[-1].v0,
                max_vel=max_vel,
                max_acc=max_acc))

        can_reach, parts, reason = BangBangTrajectory1D.canReach(initial_pos, final_pos, initial_vel, max_vel, max_acc,
                                                                 target_time)

        if not can_reach:
            return finish_up(parts)
        else:
            shortest = BangBangTrajectory1D.generateShortest(initial_pos, final_pos, initial_vel, max_vel, max_acc)
            if shortest[-1].t_end + CONSTANT_SPEED_DURATION / 2 - ACCURACY < target_time:
                return shortest
            BangBangTrajectory1D.slowDownFastest(parts, final_pos, max_vel, max_acc, target_time)
            return finish_up(parts)

    @staticmethod
    def checkAndCombineParts(original: List[BBTrajectoryPart], parts: List[BBTrajectoryPart]) -> List[BBTrajectoryPart]:
        combined = list(original)
        if len(original) > 1:
            last_part = original[-1]
            last_last_t_end = original[-2].t_end
            time_offset = last_part.t_end
        elif len(original) > 0:
            last_part = original[-1]
            last_last_t_end = 0.0
            time_offset = last_part.t_end
        else:
            last_part = None
            last_last_t_end = 0.0
            time_offset = 0.0

        for part in parts:
            current_part = BBTrajectoryPart()
            current_part.v0 = part.v0
            current_part.s0 = part.s0
            current_part.acc = part.acc
            current_part.t_end = part.t_end + time_offset
            if last_part is not None:
                t_diff = last_part.t_end - last_last_t_end
                v1 = last_part.v0 + last_part.acc * t_diff
                s1 = last_part.s0 + 0.5 * (last_part.v0 + v1) * t_diff
                assert not math.isclose(last_part.acc, current_part.acc)
                assert t_diff >= 0, "t_diff >= 0"
                assert math.isclose(v1, current_part.v0), "{} != {}".format(v1, current_part.v0)
                assert math.isclose(s1, current_part.s0), "{} != {}".format(s1, current_part.s0)
            combined.append(current_part)
            last_last_t_end = last_part.t_end if last_part is not None else 0.0
            last_part = current_part
        return combined

    @staticmethod
    def velChangeToZero(s0: float, v0: float, aMax: float) -> float:
        a: float
        if 0 >= v0:
            a = aMax
        else:
            a = -aMax

        t = -v0 / a
        return s0 + (0.5 * v0 * t)

    @staticmethod
    def velTriToZero(s0: float, v0: float, v1: float, aMax: float):
        a1: float
        a2: float
        if v1 >= v0:
            a1 = aMax
            a2 = -aMax
        else:
            a1 = -aMax
            a2 = aMax

        t1 = (v1 - v0) / a1
        s1 = s0 + (0.5 * (v0 + v1) * t1)

        t2 = -v1 / a2
        return s1 + (0.5 * v1 * t2)

    @staticmethod
    def canReach(initial_pos: float, final_pos: float, initial_vel: float, max_vel: float, max_acc: float,
                 target_time: float) -> (bool, List[BBTrajectoryPart], str):
        fastest_direct = BangBangTrajectory1D.calcFastestDirect(initial_pos, final_pos, initial_vel, max_vel, max_acc)
        if fastest_direct[-1].t_end > target_time:
            return False, fastest_direct, "too-slow"
        slowest_direct = BangBangTrajectory1D.calcSlowestDirect(initial_pos, final_pos, initial_vel, max_acc)
        if slowest_direct is not None:
            if target_time - ACCURACY < slowest_direct[-1].t_end:
                return True, fastest_direct, "direct-slow"
            fastest_overshot = BangBangTrajectory1D.calcFastestOvershot(initial_pos, final_pos, initial_vel, max_vel,
                                                                        max_acc)

            if fastest_overshot[-1].t_end < target_time + ACCURACY:
                return True, fastest_overshot, "overshot"
            else:
                return False, fastest_overshot, "too-fast"

        return True, fastest_direct, "direct-fast"

    @staticmethod
    def appendConstantSpeedPart(parts: List[BBTrajectoryPart]):
        if len(parts) == 0:
            raise NotImplementedError
        if math.isclose(parts[-1].acc, 0):
            parts[-1].t_end += CONSTANT_SPEED_DURATION / 2
        else:
            last = parts[-1]
            t_diff = last.t_end - parts[-2].t_end if len(parts) >= 2 else last.t_end
            new_part = BBTrajectoryPart()
            new_part.v0 = last.v0 + last.acc * t_diff
            new_part.s0 = last.s0 + 0.5 * (last.v0 + new_part.v0) * t_diff
            new_part.acc = 0.0
            new_part.t_end = last.t_end + CONSTANT_SPEED_DURATION / 2
            parts.append(new_part)

    @staticmethod
    def slowDownFastest(parts: List[BBTrajectoryPart], final_pos: float, max_vel: float, max_acc: float,
                        target_time: float):
        if len(parts) == 1:
            pass
        elif len(parts) == 2:
            assert math.isclose(parts[-1].acc, 0.0)
            # https://www.wolframalpha.com/input?i=solve+v_0*t_1+%3Dv_0*t_2%2B1%2F2*a*Power%5Bt_2%2C2%5D%2Bv_1*t_3%2C+v_1+%3D+v_0%2Ba*t_2%2C+t_1%2Bt%3Dt_2%2Bt_3+for+v_1%2Ct_1%2C++t_2
            t3 = CONSTANT_SPEED_DURATION / 2
            v0 = parts[-1].v0
            a = math.copysign(max_acc, -v0)
            t = target_time - parts[-1].t_end

            sqrt = math.sqrt(a * ((a * (t3 ** 2)) - (2 * t * v0)))

            t1a = - sqrt / a - t
            t1b = sqrt / a - t
            t2a = - sqrt / a - t3
            t2b = sqrt / a - t3

            t1 = max(t1a, t1b)
            t2 = max(t2a, t2b)

            s_constant = v0 * t1
            v1 = v0 + a * t2
            s_dec = v0 * t2 + 0.5 * a * t2 ** 2
            s_slow = v1 * t3
            s_adapted = s_dec + s_slow

            assert math.isclose(t1 + t, t2 + t3)
            assert math.isclose(s_constant, s_adapted)

            if parts[0].t_end < parts[1].t_end - t1:
                parts[1].t_end = parts[1].t_end - t1

                parts.append(BBTrajectoryPart())
                t_diff = parts[1].t_end - parts[0].t_end
                parts[2].t_end = parts[1].t_end + t2
                parts[2].acc = a
                parts[2].v0 = parts[1].v0
                parts[2].s0 = t_diff * v0

                parts.append(BBTrajectoryPart())
                t_diff = parts[2].t_end - parts[1].t_end
                parts[3].t_end = parts[2].t_end + t3
                parts[3].acc = 0.0
                parts[3].v0 = parts[2].v0 + a * t_diff
                parts[3].s0 = parts[2].s0 + 0.5 * (parts[2].v0 + parts[3].v0) * t_diff
            else:
                pass
        else:
            raise NotImplementedError

    @staticmethod
    def calcFastestDirect(initial_pos: float, final_pos: float, initial_vel: float, max_vel: float, max_acc: float) \
            -> List[BBTrajectoryPart]:

        distance = final_pos - initial_pos
        a_acc = math.copysign(max_acc, distance)
        v1 = math.copysign(max_vel, distance)
        t_acc = (v1 - initial_vel) / a_acc
        assert t_acc >= 0
        s_offset_acc = 0.5 * (v1 + initial_vel) * t_acc
        if math.fabs(s_offset_acc) < math.fabs(distance):
            # Got enough space to fully accelerate
            parts = [BBTrajectoryPart(), BBTrajectoryPart()]
            parts[0].s0 = initial_pos
            parts[0].v0 = initial_vel
            parts[0].acc = a_acc
            parts[0].t_end = t_acc
            parts[1].s0 = initial_pos + s_offset_acc
            parts[1].v0 = v1
            parts[1].acc = 0
            parts[1].t_end = parts[0].t_end + math.fabs(math.fabs(s_offset_acc) - math.fabs(distance)) / math.fabs(v1)
            assert parts[0].t_end >= 0
            assert parts[1].t_end >= 0
            return parts

        sqrt = math.sqrt(2 * a_acc * distance + initial_vel ** 2)
        t1 = -(sqrt + initial_vel) / a_acc
        t2 = (sqrt - initial_vel) / a_acc

        t = t1 if distance < 0 else t2

        part = BBTrajectoryPart()
        part.s0 = initial_pos
        part.v0 = initial_vel
        part.acc = a_acc
        part.t_end = t
        assert part.t_end >= 0, "{} >= 0".format(t)
        return [part]

    @staticmethod
    def calcSlowestDirect(initial_pos: float, final_pos: float, initial_vel: float, max_acc: float) \
            -> Optional[List[BBTrajectoryPart]]:
        distance = final_pos - initial_pos

        # Detect configurations where overshoot is impossible, so the slowest direct is always as slow as we need
        if math.isclose(initial_vel, 0):
            return None
        if (distance >= 0) != (initial_vel >= 0):
            #    Init     Final
            #     |        |
            #     V        V
            #    <------ initVel
            #    distance ----->
            return None

        a_dec = math.copysign(max_acc, -initial_vel)
        t_dec = -initial_vel / a_dec
        assert t_dec >= 0
        s_offset_dec = 0.5 * initial_vel * t_dec
        if math.fabs(s_offset_dec) < math.fabs(distance):
            # Got enough space to fully stop
            return None
        part = BBTrajectoryPart()
        part.s0 = initial_pos
        part.v0 = initial_vel
        part.acc = a_dec
        part.t_end = (math.fabs(initial_vel) - math.sqrt(2 * a_dec * distance + initial_vel ** 2)) / max_acc
        return [part]

    @staticmethod
    def calcFastestOvershot(initial_pos: float, final_pos: float, initial_vel: float, max_vel: float, max_acc: float) \
            -> List[BBTrajectoryPart]:
        a_dec = math.copysign(max_acc, -initial_vel)
        t_dec = - initial_vel / a_dec
        assert t_dec >= 0
        s_offset_dec = 0.5 * initial_vel * t_dec

        parts = BangBangTrajectory1D.calcFastestDirect(initial_pos + s_offset_dec, final_pos, 0.0, max_vel, max_acc)

        parts[0].s0 = initial_pos
        parts[0].v0 = initial_vel
        assert parts[0].acc == a_dec
        for part in parts:
            part.t_end += t_dec

        return parts

    @staticmethod
    def calcTri(s0: float, v0: float, s2: float, a: float) -> List[BBTrajectoryPart]:
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

        parts = [BBTrajectoryPart(), BBTrajectoryPart()]
        parts[0].t_end = t1
        parts[0].acc = a
        parts[0].v0 = v0
        parts[0].s0 = s0
        parts[1].t_end = t1 + t2
        parts[1].acc = -a
        parts[1].v0 = v1
        parts[1].s0 = s1
        return parts

    @staticmethod
    def calcTrapz(s0: float, v0: float, v1: float, s3: float, aMax: float) -> List[BBTrajectoryPart]:
        a1: float
        a3: float
        t1: float
        t2: float
        t3: float
        v2: float
        s1: float
        s2: float

        if v0 > v1:
            a1 = -aMax
        else:
            a1 = aMax

        if v1 > 0:
            a3 = -aMax
        else:
            a3 = aMax

        t1 = (v1 - v0) / a1
        v2 = v1
        t3 = -v2 / a3

        s1 = s0 + (0.5 * (v0 + v1) * t1)
        s2 = s3 - (0.5 * v2 * t3)
        t2 = (s2 - s1) / v1

        parts = [BBTrajectoryPart(), BBTrajectoryPart(), BBTrajectoryPart()]
        parts[0].t_end = t1
        parts[0].acc = a1
        parts[0].v0 = v0
        parts[0].s0 = s0
        parts[1].t_end = t1 + t2
        parts[1].acc = 0
        parts[1].v0 = v1
        parts[1].s0 = s1
        parts[2].t_end = t1 + t2 + t3
        parts[2].acc = a3
        parts[2].v0 = v2
        parts[2].s0 = s2
        return parts
