from __future__ import annotations

from dataclasses import dataclass
from typing import List

from oop.gadgets.gadgetlike import GadgetLike
from oop.gadgets.gadgetdefs import (
    AntiParallel2Toggle,
    Crossing2Toggle,
    NoncrossingWireToggle,
    Parallel2Toggle,
)


@dataclass
class TaskConfig:
    name: str
    initial_gadgets: List[GadgetLike]
    target_gadget: GadgetLike
    max_steps: int = 8


TASK_CONFIGS = {
    "AP2T_to_C2T": TaskConfig(
        name="AP2T_to_C2T",
        initial_gadgets=[AntiParallel2Toggle(), AntiParallel2Toggle()],
        target_gadget=Crossing2Toggle(),
    ),
    "C2T_to_AP2T": TaskConfig(
        name="C2T_to_AP2T",
        initial_gadgets=[Crossing2Toggle(), Crossing2Toggle()],
        target_gadget=AntiParallel2Toggle(),
    ),
    "C2T_to_P2T": TaskConfig(
        name="C2T_to_P2T",
        initial_gadgets=[Crossing2Toggle(), Crossing2Toggle()],
        target_gadget=Parallel2Toggle(),
    ),
    "NWT_to_AP2T": TaskConfig(
        name="NWT_to_AP2T",
        initial_gadgets=[NoncrossingWireToggle(), NoncrossingWireToggle()],
        target_gadget=AntiParallel2Toggle(),
    ),
}
