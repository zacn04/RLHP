from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

from stable_baselines3.common.callbacks import EvalCallback

from .envs import decode_action
from .tasks import TaskConfig
from .utils import unwrap_obs


class SuccessEvalCallback(EvalCallback):
    """Evaluation callback that logs success metrics and traces."""

    def __init__(
        self,
        eval_env_fn: Callable[[], object],
        task_cfg: TaskConfig,
        csv_path: str,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 10,
        verbose: int = 1,
    ) -> None:
        super().__init__(eval_env_fn(), eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, verbose=verbose)
        self.eval_env_fn = eval_env_fn
        self.task_cfg = task_cfg
        self.csv_path = Path(csv_path)
        self.steps_since_last_log = 0
        self.action_counts = {k: 0 for k in ["COMBINE", "CONNECT", "SET_STATE", "STOP", "DELETE"]}
        Path("traces").mkdir(exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step",
                "success_rate",
                "avg_reward",
                "illegal_actions",
                "illegal_action_rate",
                "combine_rate",
                "connect_rate",
                "set_state_rate",
                "stop_rate",
            ])

    def _get_action_type(self, action, env):
        if action == env.action_space.n - 1:
            return "STOP"
        if action < env.num_combine_ops:
            return "COMBINE"
        if action < env.num_combine_ops + env.num_connect_ops:
            return "CONNECT"
        if action < env.num_combine_ops + env.num_connect_ops + env.delete_ops:
            return "DELETE"
        return "SET_STATE"

    def _on_step(self) -> bool:
        self.steps_since_last_log += 1
        if self.steps_since_last_log >= 1000:
            illegal_actions = self.training_env.get_attr("illegal_actions")[0]
            illegal_rate = illegal_actions / self.steps_since_last_log
            total_actions = sum(self.action_counts.values())
            if total_actions > 0:
                action_props = {k: v / total_actions for k, v in self.action_counts.items()}
                logging.info(
                    "Training stats (last %s steps):", f"{self.steps_since_last_log:,}"
                )
                logging.info("  Illegal actions: %s (%.2f%%)", illegal_actions, illegal_rate * 100)
                logging.info("  Action proportions:")
                for action_type, prop in action_props.items():
                    logging.info("    %s: %.2f%%", action_type, prop * 100)
            self.steps_since_last_log = 0
            self.action_counts = {k: 0 for k in self.action_counts}

        if self.n_calls % self.eval_freq != 0:
            return True

        successes, total_reward = 0, 0.0
        first_trajectory = []
        eval_action_counts = {k: 0 for k in self.action_counts}

        for ep in range(self.n_eval_episodes):
            eval_env = self.eval_env_fn()
            obs, _ = eval_env.reset()
            obs = unwrap_obs(obs)
            done, truncated = False, False
            ep_reward = 0.0
            trajectory = []
            while not (done or truncated):
                mask = unwrap_obs(obs)["action_mask"]
                action, _ = self.model.predict(obs, deterministic=True, action_masks=mask)
                decoded = decode_action(action, eval_env.env)
                trajectory.append((decoded,))
                obs, reward, done, truncated, _ = eval_env.step(action)
                obs = unwrap_obs(obs)
                ep_reward += float(reward)
                action_type = self._get_action_type(action, eval_env.env)
                eval_action_counts[action_type] += 1
                trajectory[-1] = (decoded, float(reward))
            final_gadget = eval_env.env.network.simplify()
            target_gadget = self.task_cfg.target_gadget
            if ep == 0:
                first_trajectory = trajectory
            if final_gadget == target_gadget:
                successes += 1
            total_reward += ep_reward

        success_rate = successes / self.n_eval_episodes
        avg_reward = total_reward / self.n_eval_episodes
        illegal_actions = self.training_env.get_attr("illegal_actions")[0]
        illegal_rate = illegal_actions / self.eval_freq if self.eval_freq > 0 else 0
        total_eval_actions = sum(eval_action_counts.values())
        eval_action_props = {
            k: v / total_eval_actions if total_eval_actions > 0 else 0 for k, v in eval_action_counts.items()
        }
        print(f"\n[Eval] step={self.num_timesteps:,}")
        print("First episode trajectory:")
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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_path = Path("traces") / f"trace_{timestamp}_step{self.num_timesteps}.txt"
        with trace_path.open("w") as f:
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Step: {self.num_timesteps}\n")
            f.write("Trajectory:\n")
            for action, reward in first_trajectory:
                f.write(f"  {action} → reward={reward:.1f}\n")
            f.write(f"Final gadget: {final_gadget}\n")
            f.write(f"Target gadget: {target_gadget}\n")
            f.write(f"Success rate: {success_rate:.2%}\n")
            f.write(f"Average reward: {avg_reward:.1f}\n")
            f.write(
                f"Illegal actions (last {self.eval_freq:,} steps): {illegal_actions} ({illegal_rate:.2%})\n"
            )
            f.write("Action proportions:\n")
            for action_type, prop in eval_action_props.items():
                f.write(f"  {action_type}: {prop:.2%}\n")
        self.logger.record("eval/success_rate", success_rate)
        self.logger.record("eval/avg_reward", avg_reward)
        self.logger.record("eval/illegal_actions", illegal_actions)
        self.logger.record("eval/illegal_action_rate", illegal_rate)
        for action_type, prop in eval_action_props.items():
            self.logger.record(f"eval/{action_type.lower()}_rate", prop)

        with self.csv_path.open("a", newline="") as f:
            csv.writer(f).writerow(
                [
                    self.num_timesteps,
                    success_rate,
                    avg_reward,
                    illegal_actions,
                    illegal_rate,
                    eval_action_props["COMBINE"],
                    eval_action_props["CONNECT"],
                    eval_action_props["SET_STATE"],
                    eval_action_props["STOP"],
                ]
            )
        return True
