#!/usr/bin/env python3
import os, sys, numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from env import GadgetSimulationEnv
from oop.gadgets.gadgetdefs import *
from train import decode_action

def mask_fn(env):
    mask = env._build_action_mask()
    mask[-1] = 1        
    return mask

# Re-create the same task you trained on
env = GadgetSimulationEnv(
    initial_gadgets=[Crossing2Toggle(), Crossing2Toggle()],
    target_gadget=AntiParallel2Toggle(),
    max_steps=8,
)
env = ActionMasker(env, mask_fn)

model_path = "models/mppo_multi_20250521_175324"   
model = MaskablePPO.load(model_path, env=env)

# Run one episode
obs, _ = env.reset()
done = False
while not done:
    mask = obs["action_mask"]
    action, _ = model.predict(obs, deterministic=True, action_masks=mask)
    obs, reward, done, _, _ = env.step(action)
    print(f"action={decode_action(action, env.env)} reward={reward}")
