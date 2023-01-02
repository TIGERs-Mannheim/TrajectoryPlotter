from multiprocessing import Pool
from typing import Tuple

import tqdm

from JavaClones.Vec2 import Vec2
from Plotting.PlotType import PlotType
from Plotting.Plotter import Plotter

v_max = 3.0
a_max = 2.0

test_values = [
    ((-1.5, -3), (2, 0), 2.3),
    ((1, 0.5), (0, 0), 1),
    ((-1.5, -3), (0, 0), 2.2),
    ((-1, 0.5), (0, 0), 1),
    ((-1.5, -3), (0, 0), 2.6),
    ((-1.5, -3), (0, 0), 2.5),
    ((-1.5, -3), (0, 0), 2.4),
    ((-1.5, -3), (0, 0), 2.3),
]


def run(value: Tuple[Tuple[float, float], Tuple[float, float], float]):
    Plotter.plot(
        s0=0,
        s1=Vec2(*value[0]),
        v0=Vec2(*value[1]),
        v_max=v_max,
        a_max=a_max,
        tt=value[2],
        primary_direction=None,
        plot_type=PlotType.SIM_TRAJ,
        save_fig=True,
        show_fig=False
    )


if __name__ == '__main__':
    pool = Pool()
    for _ in tqdm.tqdm(pool.imap_unordered(run, test_values), total=len(test_values)):
        pass
