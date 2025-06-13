from __future__ import annotations

from pathlib import Path
from typing import Callable, List

import gymnasium as gym
import numpy as np
from stable_baselines3.common.wrappers import ActionMasker
import random

from rl.env import GadgetSimulationEnv
from .tasks import TaskConfig, TASK_CONFIGS


def mask_fn(env: GadgetSimulationEnv) -> np.ndarray:
    """Return the action mask for the current env state."""
    mask = env._build_action_mask()
    mask[-1] = 1
    assert mask.any(), "Action mask is all-False!"
    return mask


class MultiTaskEnv(gym.Env):
    """Cycle through a list of tasks each episode."""

    def __init__(self, task_names: List[str], freq_weighted: bool = False) -> None:
        super().__init__()
        self.task_names = task_names
        self.current_task_idx = 0
        self.env: gym.Env | None = None
        self.illegal_actions = 0
        self.freq_weighted = freq_weighted
        self._create_env()

        self.observation_space = self.env.observation_space  # type: ignore
        self.action_space = self.env.action_space  # type: ignore

    def _create_env(self) -> None:
        task = self.task_names[self.current_task_idx]
        cfg = TASK_CONFIGS[task]
        self.env = ActionMasker(
            GadgetSimulationEnv(
                initial_gadgets=cfg.initial_gadgets,
                target_gadget=cfg.target_gadget,
                max_steps=cfg.max_steps,
                freq_weighted_rewards=self.freq_weighted,
            ),
            mask_fn,
        )
        self.illegal_actions = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)  # type: ignore
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)  # type: ignore
        self.illegal_actions = self.env.env.illegal_actions  # type: ignore
        if done or truncated:
            self.current_task_idx = (self.current_task_idx + 1) % len(self.task_names)
            self._create_env()
        return obs, reward, done, truncated, info

    def get_attr(self, attr):
        if attr == "illegal_actions":
            return self.illegal_actions
        return getattr(self.env, attr)

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()  # type: ignore

    def _build_action_mask(self) -> np.ndarray:
        return self.env._build_action_mask()  # type: ignore


class RandomMultiTaskEnv(gym.Env):
    """Select a random task each episode."""

    def __init__(self, task_names: List[str], freq_weighted: bool = False) -> None:
        super().__init__()
        self.task_names = task_names
        self.env: gym.Env | None = None
        self.illegal_actions = 0
        self.freq_weighted = freq_weighted
        self._create_env()

        self.observation_space = self.env.observation_space  # type: ignore
        self.action_space = self.env.action_space  # type: ignore

    def _create_env(self) -> None:
        task = random.choice(self.task_names)
        cfg = TASK_CONFIGS[task]
        self.env = ActionMasker(
            GadgetSimulationEnv(
                initial_gadgets=cfg.initial_gadgets,
                target_gadget=cfg.target_gadget,
                max_steps=cfg.max_steps,
                freq_weighted_rewards=self.freq_weighted,
            ),
            mask_fn,
        )
        self.illegal_actions = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)  # type: ignore
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)  # type: ignore
        self.illegal_actions = self.env.env.illegal_actions  # type: ignore
        if done or truncated:
            self._create_env()
        return obs, reward, done, truncated, info

    def get_attr(self, attr):
        if attr == "illegal_actions":
            return self.illegal_actions
        return getattr(self.env, attr)

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()  # type: ignore

    def _build_action_mask(self) -> np.ndarray:
        return self.env._build_action_mask()  # type: ignore

def make_env(task_cfg: TaskConfig, freq_weighted: bool = False) -> Callable[[], gym.Env]:
    def _make() -> gym.Env:
        env = GadgetSimulationEnv(
            initial_gadgets=task_cfg.initial_gadgets,
            target_gadget=task_cfg.target_gadget,
            max_steps=task_cfg.max_steps,
            freq_weighted_rewards=freq_weighted,
        )
        return ActionMasker(env, mask_fn)

    return _make


def make_multitask_env(task_names: List[str], freq_weighted: bool = False) -> Callable[[], gym.Env]:
    def _make() -> gym.Env:
        return MultiTaskEnv(task_names, freq_weighted)

    return _make


def make_random_multitask_env(task_names: List[str], freq_weighted: bool = False) -> Callable[[], gym.Env]:
    def _make() -> gym.Env:
        return RandomMultiTaskEnv(task_names, freq_weighted)

    return _make


def decode_action(action: int, env: GadgetSimulationEnv) -> str:
    """Decode an action index into a human-readable string."""
    if action == env.action_space.n - 1:
        return "STOP"
    if action < env.num_combine_ops:
        flat = action
        ij = flat // (4 * env.base_ports)
        rem = flat % (4 * env.base_ports)
        rot = rem // env.base_ports
        splice = rem % env.base_ports
        i = ij // env.max_gadgets
        j = ij % env.max_gadgets
        return f"COMBINE(g{i}, g{j}, rot={rot}, splice={splice})"
    if action < env.num_combine_ops + env.num_connect_ops:
        conn_idx = action - env.num_combine_ops
        g_idx = conn_idx // (env.max_ports * env.max_ports)
        rem = conn_idx % (env.max_ports * env.max_ports)
        loc1 = rem // env.max_ports
        loc2 = rem % env.max_ports
        if g_idx < len(env.network.subgadgets):
            locs = env.network.subgadgets[g_idx].getLocations()
            lbl1 = locs[loc1] if loc1 < len(locs) else "⟂"
            lbl2 = locs[loc2] if loc2 < len(locs) else "⟂"
        else:
            lbl1 = lbl2 = "⟂"
        return f"CONNECT(g{g_idx}, {lbl1}, {lbl2})"
    if action < env.num_combine_ops + env.num_connect_ops + env.num_setstate_ops + env.num_delete_ops:
        del_idx = action - env.num_combine_ops - env.num_connect_ops - env.num_setstate_ops
        g_idx = del_idx // env.max_ports
        loc = del_idx % env.max_ports
        g = env.network.subgadgets[g_idx]
        port_label = g.getLocations()[loc]
        return f"DELETE(loc {port_label} of g{g_idx})"
    set_idx = action - env.num_combine_ops - env.num_connect_ops
    g_idx = set_idx // env.max_states
    s = set_idx % env.max_states
    return f"SET_STATE(g{g_idx}, s={s})"
