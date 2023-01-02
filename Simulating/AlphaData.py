from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class AlphaData:
    optimal: float
    alphas: List[float]
    diffs: List[float]
    x_times: List[float]
    y_times: List[float]

    def __post_init__(self):
        assert len(self.alphas) == len(self.diffs) == len(self.x_times) == len(self.y_times)
