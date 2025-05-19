import os
import sys
from tqdm import tqdm
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
from gymnasium import spaces

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# Import your gadget classes and env
from oop.gadgets.gadgetdefs import (
    AntiParallel2Toggle, Crossing2Toggle, Toggle2,
    AntiParallelLocking2Toggle, CrossingLocking2Toggle, ParallelLocking2Toggle,
    Door, SelfClosingDoor
)
from rl.search.env import GadgetSimulationEnv
from rl.search.exhaustive.search import format_operation, get_possible_operations

def plot_training_progress(rewards, title):
    plt.figure(figsize=(10, 5))
    # Plot raw rewards
    plt.plot(rewards, alpha=0.3, label='Raw Rewards')
    # Plot moving average
    window_size = min(10, len(rewards))
    if window_size > 0:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), moving_avg, label=f'{window_size}-Episode Moving Average')
    plt.title(f'Training Progress - {title}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'training_{title.lower().replace(" ", "_")}.png')
    plt.close()

def run_rl_simulation(
    initial_gadgets,
    target_gadget,
    title="RL Simulation",
    max_steps=8,
    dqn_params=None,
    seed_trajectories=None,      # list of expert op‐lists, e.g. `[ [("COMBINE",...),...,("STOP",)] ]`
    total_timesteps=10_000,       # overall training budget
    eval_episodes=50,             # for final deterministic evaluation
    verbose=True,
):
    """
    Runs a DQN agent to build target_gadget from initial_gadgets.
    - Seeds replay buffer with seed_trajectories if provided.
    - Trains for total_timesteps with prioritized replay.
    - Evaluates over eval_episodes deterministic rollouts.
    Returns (model, success_rate).
    """
    # 1) Build and wrap environments
    train_env = GadgetSimulationEnv(initial_gadgets, target_gadget, max_steps=max_steps)
    check_env(train_env)
    venv = DummyVecEnv([lambda: GadgetSimulationEnv(initial_gadgets, target_gadget, max_steps=max_steps)])

    # 2) DQN hyperparams (override via dqn_params)
    default_dqn = dict(
        policy="MultiInputPolicy",
        learning_rate=1e-4,
        buffer_size=5_000,
        learning_starts=1,
        batch_size=32,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        tau=0.1,
        gamma=0.99,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        prioritized_replay=True,
        prioritized_replay_alpha=0.6,
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=0,
    )
    if dqn_params:
        default_dqn.update(dqn_params)

    model = DQN(env=venv, **default_dqn)

    # 3) Seed replay buffer with expert demos
    if seed_trajectories:
        for demo in seed_trajectories:
            obs = venv.reset()[0]
            done = False
            for op in demo:
                possible_ops = get_possible_operations(train_env.network)
                if op not in possible_ops:
                    break
                idx = possible_ops.index(op)
                next_obs, reward, done, _, _ = venv.step([idx])
                model.replay_buffer.add(
                    obs=obs,
                    next_obs=next_obs,
                    action=idx,
                    reward=reward,
                    done=done,
                    infos={},
                )
                obs = next_obs
                if done:
                    break

    # 4) Train for a fixed timestep budget
    pbar = tqdm(total=total_timesteps, desc=f"Training {title}")
    timesteps = 0
    while timesteps < total_timesteps:
        # train_freq steps per batch
        model.learn(total_timesteps=default_dqn["train_freq"],
                    reset_num_timesteps=False)
        timesteps += default_dqn["train_freq"]
        pbar.update(default_dqn["train_freq"])
    pbar.close()

    # 5) Final deterministic evaluation
    successes = 0
    for _ in range(eval_episodes):
        obs = venv.reset()[0]
        done = False
        steps = 0
        # reset the env’s network if needed
        train_env.network = deepcopy(train_env.network)
        train_env.operation_history.clear()

        while not done and steps < max_steps:
            possible_ops = get_possible_operations(train_env.network)
            if not possible_ops:
                break
            action, _ = model.predict(obs, deterministic=True)
            action = action[0]
            if action >= len(possible_ops):
                action = np.random.randint(len(possible_ops))
            obs, reward, done, _, _ = venv.step([action])
            steps += 1

        # after episode, check if final gadget matches
        final = train_env.network.simplify()
        if final == target_gadget:
            successes += 1

    success_rate = successes / eval_episodes
    if verbose:
        print(f"\nTest result for {title}: {success_rate:.2%} success over {eval_episodes} runs")

    return model, success_rate >= 1.0


########################
# Define your tests here
########################

RL_TESTS = {
    "AP2T -> C2T": {
        "initial_gadgets": [AntiParallel2Toggle(), AntiParallel2Toggle()],
        "target_gadget": Crossing2Toggle(),
        "dqn_params": {
            "learning_rate": 0.0001,
            "exploration_fraction": 0.4,  # More exploration for this complex transformation
        },
    },
    "CL2T -> PL2T": {
        "initial_gadgets": [CrossingLocking2Toggle(), CrossingLocking2Toggle()],
        "target_gadget": ParallelLocking2Toggle(),
        "dqn_params": {
            "learning_rate": 0.0001,
            "exploration_fraction": 0.4,
        },
    },
    "Door -> SelfClosingDoor": {
        "initial_gadgets": [Door(), Door()],
        "target_gadget": SelfClosingDoor(),
        "dqn_params": {
            "learning_rate": 0.0001,
            "exploration_fraction": 0.3,
        },
    },
    "C2T -> Toggle2": {
        "initial_gadgets": [Crossing2Toggle(), Crossing2Toggle()],
        "target_gadget": Toggle2(),
        "dqn_params": {
            "learning_rate": 0.0001,
            "exploration_fraction": 0.3,
        },
    },
    # Add more tests here easily!
}

def run_all_rl_tests(selected_tests=None, **override_params):
    tests = RL_TESTS if selected_tests is None else {k: RL_TESTS[k] for k in selected_tests}
    results = {}
    for test_name, params in tests.items():
        print(f"\nRunning test: {test_name}")
        config = dict(
            max_episodes=params.get("max_episodes", 1000),  # Increased default
            train_timesteps_per_ep=params.get("train_timesteps_per_ep", 2048),  # Increased default
            min_success_rate=0.1,  # New default
            max_episodes_without_success=200,  # New default
        )
        config.update({k: v for k, v in params.items() if k not in {"initial_gadgets", "target_gadget"}})
        config.update(override_params)  # For user override
        _, passed = run_rl_simulation(
            initial_gadgets=params["initial_gadgets"],
            target_gadget=params["target_gadget"],
            title=test_name,
            **config
        )
        results[test_name] = passed
        print(f"{test_name}: {'✅ PASSED' if passed else '❌ FAILED'}")
    print("\n=== Test Summary ===")
    print("Total:", len(results), "Passed:", sum(results.values()), "Failed:", len(results) - sum(results.values()))
    return all(results.values())

if __name__ == "__main__":
    # Example usage: run_all_rl_tests()
    #run_all_rl_tests()
    
    expert_demo = [
    ("COMBINE", 0, 1, 1, 0),
    ("CONNECT", 0, 1, 2),
    ("CONNECT", 0, 2, 5),
    ("STOP",)
    ]

    model, rate = run_rl_simulation(
        initial_gadgets=[AntiParallel2Toggle(), AntiParallel2Toggle()],
        target_gadget=Crossing2Toggle(),
        title="AP2T -> C2T",
        seed_trajectories=[expert_demo],
        total_timesteps=5000,
        eval_episodes=50,
    )

