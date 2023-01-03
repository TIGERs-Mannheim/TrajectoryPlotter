from multiprocessing import Pool
from typing import Tuple, Union

import tqdm

from JavaClones.Vec2 import Vec2
from Plotting.PlotType import PlotType
from Plotting.Plotter import Plotter

v_max = 2.0
a_max = 3.0

test_values = [
    (Vec2(1, 0.5), Vec2(0, 0), 1),
    (Vec2(-1, 0.5), Vec2(0, 0), 1),
    (Vec2(1.5, 3), Vec2(-2, 0), 2.3),
    (Vec2(1.5, 3), Vec2(0, 0), 2.8),
    (Vec2(1.5, 3), Vec2(0, 0), 2.7),
    (Vec2(1.5, 3), Vec2(0, 0), 2.6),
    (Vec2(1.5, 3), Vec2(0, 0), 2.5),
    (Vec2(1.5, 3), Vec2(0, 0), 2.4),
    (Vec2(1.5, 3), Vec2(0, 0), 2.3),
    (Vec2(1.5, 3), Vec2(0, 0), 2.2),
    (Vec2(1.572320957798463147980783105595037341117858886718750000000000,
          -2.634918375543914059733197063906118273735046386718750000000000),
     Vec2(1.549806787024718524037325551034882664680480957031250000000000,
          0.342242976064745185738047439372166991233825683593750000000000),
     1.841288796529363835929871129337698221206665039062500000000000),

]


def run(value: Tuple[Union[Vec2, float], Union[Vec2, float], float]):
    assert isinstance(value[0], type(value[1]))
    Plotter.plot(
        s0=0,
        s1=value[0],
        v0=value[1],
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
