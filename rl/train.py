#!/usr/bin/env python3
import os
import sys
import logging
from datetime import datetime
# add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import torch

import gymnasium as gym
from gymnasium import spaces

from oop.gadgets.gadgetdefs import AntiParallel2Toggle, Crossing2Toggle
from env import GadgetSimulationEnv

from stable_baselines3.common.base_class import BaseAlgorithm
from sb3_contrib.common.wrappers import ActionMasker


def mask_fn(env):
    return env._build_action_mask()

def main():
    # Set up logging
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info("Starting training session")
    
    # 1) Create your environment
    env = DummyVecEnv([
    lambda: ActionMasker(
        GadgetSimulationEnv(
            initial_gadgets=[AntiParallel2Toggle(), AntiParallel2Toggle()],
            target_gadget=Crossing2Toggle(),
            max_steps=8,
        ),
        mask_fn
    )
    ])

    # 2) Instantiate MaskablePPO
    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=1e-3,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.05,
        policy_kwargs=dict(
            net_arch=[256, 256],
            activation_fn=torch.nn.ReLU,
        ),
        verbose=1,
        tensorboard_log="runs/mppo_ap2t_c2t",
    )

    # 3) Train
    TIMESTEPS = 500_000
    logging.info(f"Starting training for {TIMESTEPS} timesteps")
    model.learn(total_timesteps=TIMESTEPS)
    logging.info("Training completed")

    # 4) Save the model
    os.makedirs("models", exist_ok=True)
    model_path = "models/mppo_ap2t_to_c2t"
    model.save(model_path)
    logging.info(f"Model saved to {model_path}")

    # 5) Evaluate
    successes = 0
    ep_rewards = []
    logging.info("Starting evaluation")
    for ep in range(100):
        eval_env = DummyVecEnv([
            lambda: ActionMasker(
                GadgetSimulationEnv(
                    initial_gadgets=[AntiParallel2Toggle(), AntiParallel2Toggle()],
                    target_gadget=Crossing2Toggle(),
                    max_steps=8,
                ),
                mask_fn
            )
        ])
        obs = eval_env.reset()
        done = False
        total_r = 0
        while not done:
            action_mask = obs["action_mask"]
            action, _ = model.predict(obs, deterministic=True, action_masks=action_mask)
            obs, reward, done, _ = eval_env.step(action)
            total_r += reward
        ep_rewards.append(total_r)
        net = eval_env.envs[0].env
        if net.network.simplify() == Crossing2Toggle():
            successes += 1
            logging.info(f"Episode {ep}: Success!")
        else:
            logging.info(f"Episode {ep}: Failed with reward {total_r}")

    # 6) Plot episode rewards
    plt.figure(figsize=(6, 4))
    plt.plot(ep_rewards, marker='.', linestyle='-')
    plt.title("Episode Rewards (MaskablePPO)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"rewards_{timestamp}.png"))
    plt.show()

    success_rate = successes/100
    logging.info(f"✅ Success rate over 100 episodes: {success_rate:.2%}")
    print(f"✅ Success rate over 100 episodes: {success_rate:.2%}")
    logging.info(f"Log file saved to: {log_file}")

if __name__ == "__main__":
    main()
