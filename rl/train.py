import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from oop.gadgets.gadgetdefs import *
from oop.gadgets.gadgetlike import GadgetNetwork
from env import GadgetSimulationEnv  
from expert import EXPERT_SOLUTIONS



def seed_replay_buffer(model, trajs, env):
    # Reset env so obs shapes line up
    obs = env.reset()
    for demo in trajs:
        env.network = GadgetNetwork()
        for g in deepcopy(env.initial_gadgets):
            env.network += g
        for op in demo:
            # turn (COMBINE/CONNECT/STOP) into action idx
            action = env.action_from_op(op)
            next_obs, reward, done, info = env.step(action)
            model.replay_buffer.add(
                obs, next_obs, np.array([action]), [reward], [done], [info]
            )
            obs = next_obs
            if done:
                break

if __name__ == "__main__":
    # 2) Create the vectorized env
    raw_env = GadgetSimulationEnv(
        initial_gadgets=[AntiParallel2Toggle(), AntiParallel2Toggle()],
        target_gadget=Crossing2Toggle(),
        max_steps=8
    )
    env = DummyVecEnv([lambda: raw_env])

    # 3) Build the model
    model = DQN(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=10_000,
        learning_starts=0,      # we seed immediately
        batch_size=32,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[64,64]),
        verbose=1,
    )

    # 4) Seed the replay buffer with your expert demo
    seed_replay_buffer(model, EXPERT_SOLUTIONS["AP2T_to_C2T"], raw_env)

    # 5) Train
    TIMESTEPS = 20_000
    model.learn(total_timesteps=TIMESTEPS)

    # 6) Save
    os.makedirs("models", exist_ok=True)
    model.save("models/dqn_ap2t_to_c2t")

    # 7) Evaluate
    successes = 0
    EPISODES = 100
    ep_rewards = []
    for _ in range(EPISODES):
        obs = raw_env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = raw_env.step(action)
            total_reward += reward
        if raw_env.network.simplify() == Crossing2Toggle():
            successes += 1
        ep_rewards.append(total_reward)
    plt.figure()
    plt.plot(ep_rewards)
    plt.title("Rewards over Time")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()
    print(f"Success rate: {successes/EPISODES:.2%}")
