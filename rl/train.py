#!/usr/bin/env python3
"""Train MaskablePPO on the mechanical gadget tasks."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from source.callbacks import SuccessEvalCallback
from source.envs import make_env, make_multitask_env
from source.policies import RenormalizingMaskableMultiInputActorCriticPolicy
from source.tasks import TASK_CONFIGS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="AP2T_to_C2T", help="Task key or 'multi'")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--n_eval", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--freq_weighted", action="store_true", help="Use frequency-weighted rewards")
    args = parser.parse_args()

    models_dir = Path("models")
    if models_dir.exists():
        for file in models_dir.iterdir():
            if file.name.startswith("mppo_"):
                file.unlink()
    models_dir.mkdir(exist_ok=True)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{args.task}_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logging.info("Starting run with args: %s", args)

    task_names = list(TASK_CONFIGS.keys())
    train_env = DummyVecEnv([make_multitask_env(task_names, args.freq_weighted) for _ in range(8)])

    eval_task = TASK_CONFIGS[args.task]
    eval_env_fn = make_env(eval_task, args.freq_weighted)

    csv_path = log_dir / f"eval_{args.task}_{timestamp}.csv"
    eval_cb = SuccessEvalCallback(
        eval_env_fn=eval_env_fn,
        task_cfg=eval_task,
        csv_path=str(csv_path),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval,
    )

    model = MaskablePPO(
        policy=RenormalizingMaskableMultiInputActorCriticPolicy,
        env=train_env,
        learning_rate=args.lr,
        n_steps=256,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.98,
        clip_range=0.2,
        ent_coef=0.1,
        policy_kwargs=dict(net_arch=[256, 256], activation_fn=torch.nn.ReLU),
        verbose=1,
        tensorboard_log=str(Path("runs") / f"mppo_{args.task}_latest"),
    )

    logging.info("Training for %s timesteps…", f"{args.timesteps:,}")
    model.learn(total_timesteps=args.timesteps, callback=eval_cb)

    model_path = models_dir / f"mppo_{args.task}_latest"
    model.save(str(model_path))
    logging.info("Model saved → %s", model_path)


if __name__ == "__main__":
    main()
