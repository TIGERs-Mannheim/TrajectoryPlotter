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
    primary_direction: Optional[Vec2] = None

    def __post_init__(self):
        if isinstance(self.v0, (float, int)) and self.v0 == 0 and isinstance(self.s, Vec2):
            object.__setattr__(self, "v0", Vec2.zero())
        assert isinstance(self.v0, type(self.s)), "Mismatching type of distance and initial_vel"
