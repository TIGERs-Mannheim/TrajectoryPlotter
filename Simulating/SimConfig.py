from dataclasses import dataclass
from typing import Union, Optional

from JavaClones.Vec2 import Vec2


@dataclass(frozen=True)
class SimConfig:
    s: Union[float, Vec2] = 0.0
    v0: Union[float, Vec2] = 0.0
    tt: Optional[float] = None
    v_max: float = 3.0
    a_max: float = 2.0
    v_max_1d: Optional[float] = None
    a_max_1d: Optional[float] = None
    primary_direction: Optional[Vec2] = None

    def __post_init__(self):
        if isinstance(self.v0, (float, int)) and self.v0 == 0 and isinstance(self.s, Vec2):
            object.__setattr__(self, "v0", Vec2.zero())
        if isinstance(self.s, int):
            object.__setattr__(self, "s", float(self.s))
        if isinstance(self.v0, int):
            object.__setattr__(self, "v0", float(self.v0))
        if isinstance(self.tt, int):
            object.__setattr__(self, "tt", float(self.tt))
        if isinstance(self.v_max, int):
            object.__setattr__(self, "v_max", float(self.v_max))
        if isinstance(self.a_max, int):
            object.__setattr__(self, "a_max", float(self.a_max))
        if isinstance(self.v_max_1d, int):
            object.__setattr__(self, "v_max_1d", float(self.v_max_1d))
        if isinstance(self.a_max_1d, int):
            object.__setattr__(self, "a_max_1d", float(self.a_max_1d))
        assert isinstance(self.v0, type(self.s)), "Mismatching type of distance and initial_vel"
