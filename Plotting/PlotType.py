from enum import IntEnum


class PlotType(IntEnum):
    NONE = 0
    TRAJ = 1
    TRAJ_NO_ACC = 2
    SIM_TRAJ = 3
    SIM_TRAJ_NO_ACC = 4
    DIFF_ALPHA = 5
