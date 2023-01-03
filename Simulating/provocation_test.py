import random
from dataclasses import replace
from multiprocessing import Process, Value, Lock

import numpy as np

from JavaClones.BangBangTrajectoryFactory import BangBangTrajectoryFactory
from JavaClones.DestinationForTimedPositionCalc import DestinationForTimedPositionCalc
from JavaClones.Vec2 import Vec2
from Simulating.SimConfig import SimConfig
from Simulating.Simulator import Simulator

success: Value = Value("i", 0)
total: Value = Value("i", 0)
unexpected_hit: Value = Value("i", 0)
miss: Value = Value("i", 0)
total_miss: Value = Value("d", 0.0)
time_miss: Value = Value("d", 0.0)
counter_lock = Lock()


def worker_main(seed: int):
    precision = 0.01
    sim_frequency = 100
    rand = random.Random(seed)
    config = SimConfig(v_max=2.0, a_max=3.0)
    while True:
        s_x = rand.uniform(-3, 3)
        s_y = rand.uniform(-3, 3)
        s = Vec2(s_x, s_y)
        v0x = rand.uniform(-2.0, 2.0)
        v0y = rand.uniform(-2.0, 2.0)
        v0 = Vec2(v0x, v0y)
        tt = rand.uniform(0, 3)
        s_new, _ = DestinationForTimedPositionCalc().destination_for_bang_bang_2d_sync(Vec2.zero(), s, v0, 2.0, 3.0, tt)
        traj = BangBangTrajectoryFactory.traj_2d_sync(Vec2.zero(), s_new, v0, 2.0, 3.0)
        if (s - traj.get_position(tt)).get_length2() < (precision ** 2):
            expected_hit = True
        else:
            expected_hit = False
        num_steps = round(traj.get_total_time() * sim_frequency)
        config2 = replace(config, s=s, v0=v0, tt=tt)
        steps = Simulator.simulate(config2, num_steps, 1)
        did_hit = False
        closest2 = float("inf")
        closest_time = float("inf")
        for step in steps:
            distance2 = (s - step.current_pos()).get_length2()
            time_diff = step.current_time() - tt
            if distance2 < (precision ** 2):
                if abs(time_diff) < precision:
                    did_hit = True
                elif time_diff < precision and step.current_vel().get_length2() < (precision ** 2):
                    did_hit = True
                elif time_diff < closest_time:
                    closest_time = time_diff
            elif time_diff < precision and closest2 < distance2:
                closest2 = distance2

        with counter_lock:
            if expected_hit == did_hit:
                success.value += 1
            elif expected_hit:
                miss.value += 1
                total_miss.value += np.sqrt(closest2)
                time_miss.value += closest_time
                print(np.sqrt(closest2))
                print(closest_time)
                print(f"(Vec2({s_x:.60f},{s_y:.60f}),Vec2({v0x:.60f},{v0y:.60f}),{tt:.60f})  # MISS")
            else:
                unexpected_hit.value += 1
                print(f"(Vec2({s_x:.60f},{s_y:.60f}),Vec2({v0x:.60f},{v0y:.60f}),{tt:.60f})  # UNEXPECTED HIT")
            total.value += 1

            if total.value % 10 == 0:
                print(f"t {total.value:7d} | s {success.value / total.value * 100:7.3f}% | "
                      f"uh {unexpected_hit.value / total.value * 100:7.3f}% | "
                      f"m {miss.value / total.value * 100:7.3f}% | "
                      f"m dist [mm] {total_miss.value / miss.value * 1_000 if miss.value != 0 else 0:.6f} | "
                      f"m time [s] {time_miss.value / miss.value if miss.value != 0 else 0:.6f}")


if __name__ == '__main__':
    # worker_main()
    worker = []
    for i in range(16):
        worker.append(Process(target=worker_main, args=(i,)))
        print(f"Created worker {i}")
    for i in range(len(worker)):
        worker[i].start()
        print(f"Started worker {i}")
    for i in range(len(worker)):
        worker[i].join()
        print(f"Joined worker {i}")
