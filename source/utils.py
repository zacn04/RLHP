from __future__ import annotations

from typing import Any

import numpy as np


def unwrap_obs(obs: Any) -> Any:
    """Normalize observations returned by Gymnasium/DummyVecEnv."""
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, (list, np.ndarray)) and len(obs) == 1:
        obs = obs[0]
    if isinstance(obs, str):
        import ast

        try:
            obs = ast.literal_eval(obs)
        except Exception:
            pass
    return obs
