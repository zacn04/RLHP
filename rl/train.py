#!/usr/bin/env python3
"""
Train MaskablePPO (SB3-contrib) to synthesize mechanical gadgets.
Supports:
  • single-task training      →  --task AP2T_to_C2T
  • curriculum (random task) →  --task multi
Logs:
  • TensorBoard (success-rate + reward curves)
  • CSV file with eval statistics every eval_freq steps
  • Matplotlib reward plot after training
"""

import os
import sys
import csv
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import List
import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib.common.maskable.distributions import MaskableCategorical
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy

# ────────────────────────────────────────────────────────────────────────────────
# Project imports – add repo root to path so "env" and "oop" are importable.
# ────────────────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from rl.env import GadgetSimulationEnv  
from oop.gadgets.gadgetdefs import * 
from oop.gadgets.gadgetlike import GadgetLike 

# ———–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Unwrapping obs because its techy 
# ———–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

def unwrap_obs(obs):
    """
    Normalize Gymnasium / DummyVecEnv observations to a plain dict.
    Handles tuples (obs, info) and length-1 vectors.
    """
    if isinstance(obs, tuple):          # (obs, info)
        obs = obs[0]
    if isinstance(obs, (list, np.ndarray)) and len(obs) == 1:  # [obs] or array([obs], dtype=object)
        obs = obs[0]
    if isinstance(obs, str):  # Handle case where obs is converted to string
        import ast
        try:
            obs = ast.literal_eval(obs)
        except:
            pass
    return obs


# ———–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Renormalisation class bc techy  
# ———–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
class RenormalizingMaskableMultiInputActorCriticPolicy(MaskableMultiInputActorCriticPolicy):
    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> MaskableCategorical:
        dist = super()._get_action_dist_from_latent(latent_pi)
        p = dist.distribution.probs
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        dist.distribution.probs = p  
        return dist


# ────────────────────────────────────────────────────────────────────────────────
# Task specification – TBA
# ────────────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────────────
# Action-mask helper
# ────────────────────────────────────────────────────────────────────────────────
def mask_fn(env):
    """Return the action-mask for the current env state."""
    mask = env._build_action_mask()
    mask[-1] = 1
    assert mask.any(), "Action mask is all‐False!"
    return mask

# ────────────────────────────────────────────────────────────────────────────────
# Factory helpers – single task or multi-task
# ────────────────────────────────────────────────────────────────────────────────
def make_env(task_cfg: TaskConfig):
    def _make():
        env = GadgetSimulationEnv(
            initial_gadgets=task_cfg.initial_gadgets,
            target_gadget=task_cfg.target_gadget,
            max_steps=task_cfg.max_steps,
        )
        return ActionMasker(env, mask_fn)
    return _make


class MultiTaskEnv(gym.Env):
    def __init__(self, task_names: List[str]):
        super().__init__()
        self.task_names = task_names
        self.current_task_idx = 0
        self.env = None
        self.illegal_actions = 0  # Initialize illegal_actions counter
        self._create_env()
        
        # Set observation and action spaces from the underlying env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
    
    def _create_env(self):
        task = self.task_names[self.current_task_idx]
        cfg = TASK_CONFIGS[task]
        self.env = ActionMasker(
            GadgetSimulationEnv(
                initial_gadgets=cfg.initial_gadgets,
                target_gadget=cfg.target_gadget,
                max_steps=cfg.max_steps,
            ),
            mask_fn
        )
        # Reset illegal_actions when creating new env
        self.illegal_actions = 0
    
    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # Update illegal_actions from underlying env
        self.illegal_actions = self.env.env.illegal_actions  # Access through ActionMasker to base env
        if done or truncated:
            # Move to next task
            self.current_task_idx = (self.current_task_idx + 1) % len(self.task_names)
            self._create_env()
        return obs, reward, done, truncated, info
    
    def get_attr(self, attr):
        if attr == 'illegal_actions':
            return self.illegal_actions
        return getattr(self.env, attr)
        
    # Add action masking support
    def action_masks(self):
        """Return the action mask for the current state."""
        return self.env.action_masks()
        
    def _build_action_mask(self):
        """Build the action mask for the current state."""
        return self.env._build_action_mask()


def make_multitask_env(task_names: List[str]):
    def _make():
        return MultiTaskEnv(task_names)
    return _make

# ────────────────────────────────────────────────────────────────────────────────
# Evaluation callback – logs success-rate every eval_freq steps
# ────────────────────────────────────────────────────────────────────────────────
def decode_action(action, env):
    """Decode an action index into a human-readable string.
    Matches the exact action handling logic in GadgetSimulationEnv._apply_action.
    """
    if action == env.action_space.n - 1:
        return "STOP"
    elif action < env.num_combine_ops:
        # COMBINE action
        flat = action
        ij = flat // (4 * env.base_ports)
        rem = flat % (4 * env.base_ports)
        rot = rem // env.base_ports
        splice = rem % env.base_ports
        i = ij // env.max_gadgets
        j = ij % env.max_gadgets
        return f"COMBINE(g{i}, g{j}, rot={rot}, splice={splice})"
    elif action < env.num_combine_ops + env.num_connect_ops:
        # CONNECT action
        
        conn_idx = action - env.num_combine_ops
        g_idx = conn_idx // (env.max_ports * env.max_ports)
        rem = conn_idx % (env.max_ports * env.max_ports)
        loc1 = rem // env.max_ports
        loc2 = rem % env.max_ports
        if g_idx < len(env.network.subgadgets):
            locs = env.network.subgadgets[g_idx].getLocations()
            lbl = lambda k: locs[k] if k < len(locs) else "⟂"
            lbl1, lbl2 = lbl(loc1), lbl(loc2)
        else:
            lbl1 = lbl2 = "⟂"

        return (f"CONNECT(g{g_idx}, "
                f"loc{loc1}→{lbl1}, loc{loc2}→{lbl2})")
    else:
        # SET_STATE action
        set_idx = action - env.num_combine_ops - env.num_connect_ops
        g_idx = set_idx // env.max_states
        s = set_idx % env.max_states
        return f"SET_STATE(g{g_idx}, s={s})"

class SuccessEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env_fn,
        task_cfg: TaskConfig,
        csv_path: str,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 10,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.task_cfg = task_cfg
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.csv_path = csv_path
        self.steps_since_last_log = 0
        self.action_counts = {
            'COMBINE': 0,
            'CONNECT': 0,
            'SET_STATE': 0,
            'STOP': 0
        }
        # Create traces directory if it doesn't exist
        os.makedirs("traces", exist_ok=True)
        # prepare CSV
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "success_rate", "avg_reward", "illegal_actions", "illegal_action_rate",
                "combine_rate", "connect_rate", "set_state_rate", "stop_rate"
            ])

    def _get_action_type(self, action, env):
        if action == env.action_space.n - 1:
            return "STOP"
        elif action < env.num_combine_ops:
            return "COMBINE"
        elif action < env.num_combine_ops + env.num_connect_ops:
            return "CONNECT"
        else:
            return "SET_STATE"

    def _on_step(self) -> bool:
        # Track actions during training
        self.steps_since_last_log += 1
        if self.steps_since_last_log >= 1000:
            illegal_actions = self.training_env.get_attr('illegal_actions')[0]
            illegal_rate = illegal_actions / self.steps_since_last_log
            
            # Calculate action proportions
            total_actions = sum(self.action_counts.values())
            if total_actions > 0:
                action_props = {
                    k: v/total_actions for k, v in self.action_counts.items()
                }
                logging.info(f"Training stats (last {self.steps_since_last_log:,} steps):")
                logging.info(f"  Illegal actions: {illegal_actions} ({illegal_rate:.2%})")
                logging.info(f"  Action proportions:")
                for action_type, prop in action_props.items():
                    logging.info(f"    {action_type}: {prop:.2%}")
            
            self.steps_since_last_log = 0
            # Reset action counts for next period
            self.action_counts = {k: 0 for k in self.action_counts}

        if self.n_calls % self.eval_freq != 0:
            return True
        
        successes, total_reward = 0, 0.0
        first_trajectory = []  # Store the first episode's trajectory
        eval_action_counts = {k: 0 for k in self.action_counts}  # Reset for evaluation

        for ep in range(self.n_eval_episodes):
            eval_env = self.eval_env_fn()
            obs, _ = eval_env.reset()                     # ← unpack reset()
            obs = unwrap_obs(obs)
            done, truncated = False, False
            ep_reward = 0.0
            trajectory = []

            while not (done or truncated):                # ← stop on either flag
                mask   = unwrap_obs(obs)["action_mask"]               # ← easier to read
                action, _ = self.model.predict(obs,
                                            deterministic=True,
                                            action_masks=mask)
                decoded_action = decode_action(action, eval_env.env)
                trajectory.append((decoded_action,))
                obs, reward, done, truncated, _ = eval_env.step(action)
                obs = unwrap_obs(obs)
                ep_reward += float(reward)                # ← cast to float
                
                # Track action type
                action_type = self._get_action_type(action, eval_env.env)
                eval_action_counts[action_type] += 1
                
                # Decode the action and store in trajectory
                trajectory[-1] = ((decoded_action, float(reward)))

            final_gadget = eval_env.env.network.simplify()
            target_gadget = self.task_cfg.target_gadget

            # Store first episode's trajectory
            if ep == 0:
                first_trajectory = trajectory
            
            if final_gadget == target_gadget:
                print(final_gadget, final_gadget.locations, final_gadget.states, final_gadget.transitions)
                print(target_gadget, target_gadget.locations, target_gadget.states, target_gadget.transitions)
                successes += 1
            total_reward += ep_reward

        success_rate = successes / self.n_eval_episodes
        avg_reward   = total_reward / self.n_eval_episodes

        # Get current training stats
        illegal_actions = self.training_env.get_attr('illegal_actions')[0]
        illegal_rate = illegal_actions / self.eval_freq if self.eval_freq > 0 else 0

        # Calculate evaluation action proportions
        total_eval_actions = sum(eval_action_counts.values())
        eval_action_props = {
            k: v/total_eval_actions if total_eval_actions > 0 else 0 
            for k, v in eval_action_counts.items()
        }

        # Log the first episode's trajectory
        print(f"\n[Eval] step={self.num_timesteps:,}")
        print(f"First episode trajectory:")
        for action, reward in first_trajectory:
            print(f"  {action} → reward={reward:.1f}")
        print(f"Final gadget: {final_gadget}")
        print(f"Target gadget: {target_gadget}")
        print(f"Success rate: {success_rate:.2%}")
        print(f"Average reward: {avg_reward:.1f}")
        print(f"Illegal actions (last {self.eval_freq:,} steps): {illegal_actions} ({illegal_rate:.2%})")
        print("Action proportions:")
        for action_type, prop in eval_action_props.items():
            print(f"  {action_type}: {prop:.2%}")

        # Save trace to file with timestamp and timestep
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_path = os.path.join("traces", f"trace_{timestamp}_step{self.num_timesteps}.txt")
        with open(trace_path, "w") as f:
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Step: {self.num_timesteps}\n")
            f.write("Trajectory:\n")
            for action, reward in first_trajectory:
                f.write(f"  {action} → reward={reward:.1f}\n")
            f.write(f"Final gadget: {final_gadget}\n")
            f.write(f"Target gadget: {target_gadget}\n")
            f.write(f"Success rate: {success_rate:.2%}\n")
            f.write(f"Average reward: {avg_reward:.1f}\n")
            f.write(f"Illegal actions (last {self.eval_freq:,} steps): {illegal_actions} ({illegal_rate:.2%})\n")
            f.write("Action proportions:\n")
            for action_type, prop in eval_action_props.items():
                f.write(f"  {action_type}: {prop:.2%}\n")

        # tensorboard
        self.logger.record("eval/success_rate", success_rate)
        self.logger.record("eval/avg_reward", avg_reward)
        self.logger.record("eval/illegal_actions", illegal_actions)
        self.logger.record("eval/illegal_action_rate", illegal_rate)
        for action_type, prop in eval_action_props.items():
            self.logger.record(f"eval/{action_type.lower()}_rate", prop)

        # CSV
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                self.num_timesteps, 
                success_rate, 
                avg_reward,
                illegal_actions,
                illegal_rate,
                eval_action_props['COMBINE'],
                eval_action_props['CONNECT'],
                eval_action_props['SET_STATE'],
                eval_action_props['STOP']
            ])

        return True


# ────────────────────────────────────────────────────────────────────────────────
# Main – parse args, train, evaluate, plot
# ────────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="AP2T_to_C2T", help="Task key or 'multi'")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--n_eval", type=int, default=10)
    args = parser.parse_args()

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{args.task}_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    logging.info("Starting run with args: %s", args)

    # build train & eval envs
    if args.task == "multi":
        task_names = list(TASK_CONFIGS.keys())
        train_env = DummyVecEnv([make_multitask_env(task_names) for _ in range(8)])
        # For evaluation, we'll evaluate on all tasks
        eval_task = TASK_CONFIGS["AP2T_to_C2T"]  # Default eval task
    else:
        eval_task = TASK_CONFIGS[args.task]
        train_env = DummyVecEnv([make_env(eval_task) for _ in range(8)])

    eval_env_fn = make_env(eval_task)  # This returns a function that creates an env
    csv_path = os.path.join(log_dir, f"eval_{args.task}_{timestamp}.csv")
    eval_cb = SuccessEvalCallback(
        eval_env_fn=eval_env_fn,
        task_cfg=eval_task,
        csv_path=csv_path,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval,
    )

    model = MaskablePPO(
        policy=RenormalizingMaskableMultiInputActorCriticPolicy,
        env=train_env,
        learning_rate=1e-3,
        n_steps=256,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.98,
        clip_range=0.2,
        ent_coef=0.1,
        policy_kwargs=dict(
            net_arch=[256, 256], 
            activation_fn=torch.nn.ReLU, 
            ),
        verbose=1,
        tensorboard_log=os.path.join("runs", f"mppo_{args.task}_latest"),
    )

   


    logging.info("Training for %s timesteps…", f"{args.timesteps:,}")
    model.learn(total_timesteps=args.timesteps, callback=eval_cb)

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"mppo_{args.task}_latest")
    model.save(model_path)
    logging.info("Model saved → %s", model_path)


if __name__ == "__main__":
    main()
