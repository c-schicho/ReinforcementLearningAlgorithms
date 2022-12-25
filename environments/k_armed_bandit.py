from typing import Union

import numpy as np


class KArmedBandit:

    def __init__(self, k: int = 10, seed: Union[int, None] = None):
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.q_star = self.rng.normal(0, 1, self.k)

    def step(self, action: int) -> float:
        return self.rng.normal(self.q_star[action], 1)
