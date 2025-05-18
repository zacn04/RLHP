from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import numpy as np

# Import your gadget classes and env
from oop.gadgets.gadgetdefs import (
    AntiParallel2Toggle, Crossing2Toggle, Toggle2,
    AntiParallelLocking2Toggle, CrossingLocking2Toggle, ParallelLocking2Toggle,
    Door, SelfClosingDoor
)
from rl.search.env import GadgetSimulationEnv
from rl.search.exhaustive.search import format_operation

def plot_training_progress(rewards, title):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title(f'Training Progress - {title}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig(f'training_{title.lower().replace(" ", "_")}.png')
    plt.close()

def run_rl_simulation(
    initial_gadgets,
    target_gadget,
    title="RL Simulation",
    max_steps=200,
    ppo_params=None,
    max_episodes=100,
    train_timesteps_per_ep=512,
    exploration_schedule=None,
    early_stop_patience=20,
    reward_goal=100,
    negative_reward_threshold=-1000,
    verbose=True
):
    """
    Runs RL for a gadget simulation.
    All main training and evaluation options are parameterized.
    """
    # Create and check environment
    env = GadgetSimulationEnv(initial_gadgets, target_gadget, max_steps=max_steps)
    check_env(env)
    venv = DummyVecEnv([lambda: env])

    # Default PPO params
    default_ppo = dict(
        policy="MultiInputPolicy",
        learning_rate=0.0003,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.9,
        clip_range=lambda _: 0.4,
        ent_coef=0.2,
        verbose=0,
    )
    if ppo_params:
        default_ppo.update(ppo_params)
    model = PPO(env=venv, **default_ppo)

    rewards, best_reward, no_improve = [], float('-inf'), 0
    exploration_phase = 0
    if not exploration_schedule:
        # List of (patience, schedule) tuples. 
        exploration_schedule = [
            (10, lambda step, model, obs, env: model.predict(obs, deterministic=False)),
            (10, lambda step, model, obs, env: (
                [venv.action_space.sample()] if step % 5 == 0 else model.predict(obs, deterministic=False)
            )),
            (10, lambda step, model, obs, env: (
                [venv.action_space.sample()] if step % 3 == 0 else model.predict(obs, deterministic=False)
            )),
        ]

    for episode in range(max_episodes):
        model.learn(total_timesteps=train_timesteps_per_ep)
        obs = venv.reset()
        done, total_reward, steps = False, 0, 0

        # Use correct exploration schedule based on no_improve
        schedule_idx, patience_sum = 0, 0
        for patience, _ in exploration_schedule:
            patience_sum += patience
            if no_improve < patience_sum:
                break
            schedule_idx += 1
        explore_func = exploration_schedule[min(schedule_idx, len(exploration_schedule)-1)][1]

        while not done:
            action, *_ = explore_func(steps, model, obs, venv)
            obs, reward, done, info = venv.step(action)
            total_reward += reward[0]
            steps += 1

            if total_reward < negative_reward_threshold or total_reward > reward_goal:
                break

        rewards.append(total_reward)
        if verbose:
            print(f"Episode {episode+1}: reward={total_reward} steps={steps} gadgets={len(env.network.subgadgets)}")

        if total_reward > best_reward:
            best_reward = total_reward
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= early_stop_patience and episode >= early_stop_patience:
            if verbose:
                print("Early stopping due to lack of improvement.")
            break

    plot_training_progress(rewards, title)

    # Test agent
    test_env = GadgetSimulationEnv(initial_gadgets, target_gadget, max_steps=50)
    test_venv = DummyVecEnv([lambda: test_env])
    obs = test_venv.reset()
    done, total_reward, steps = False, 0, 0
    while not done and steps < 50:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_venv.step(action)
        total_reward += reward[0]
        steps += 1
        if total_reward < negative_reward_threshold or total_reward > reward_goal:
            break
        if 'error' in info[0]:
            print("Test error:", info[0]['error'])
            break

    simplified = test_env.network.simplify()
    passed = (simplified == target_gadget)
    if verbose:
        print(f"\nTest result for {title}:")
        print(f"Final total reward: {total_reward}")
        print("Proposed solution:", simplified)
        print("Target gadget:", target_gadget)
        print("✅ Passed" if passed else "❌ Failed")
        if hasattr(test_env, "successful_operations"):
            print("Operation sequence:")
            for i, op in enumerate(test_env.successful_operations, 1):
                print(f"Step {i}: {format_operation(op)}")
    return passed

########################
# Define your tests here
########################

RL_TESTS = {
    "AP2T -> C2T": {
        "initial_gadgets": [AntiParallel2Toggle(), AntiParallel2Toggle()],
        "target_gadget": Crossing2Toggle(),
        "ppo_params": {},
    },
    "CL2T -> PL2T": {
        "initial_gadgets": [CrossingLocking2Toggle(), CrossingLocking2Toggle()],
        "target_gadget": ParallelLocking2Toggle(),
        "ppo_params": {},
    },
    "Door -> SelfClosingDoor": {
        "initial_gadgets": [Door(), Door()],
        "target_gadget": SelfClosingDoor(),
        "ppo_params": {},
    },
    "C2T -> Toggle2": {
        "initial_gadgets": [Crossing2Toggle(), Crossing2Toggle()],
        "target_gadget": Toggle2(),
        "ppo_params": {"learning_rate": 0.1, "n_epochs": 20, "gamma": 0.94},
        "max_episodes": 50,
        "train_timesteps_per_ep": 1024,
    },
    # Add more tests here easily!
}

def run_all_rl_tests(selected_tests=None, **override_params):
    tests = RL_TESTS if selected_tests is None else {k: RL_TESTS[k] for k in selected_tests}
    results = {}
    for test_name, params in tests.items():
        print(f"\nRunning test: {test_name}")
        config = dict(
            max_episodes=params.get("max_episodes", 100),
            train_timesteps_per_ep=params.get("train_timesteps_per_ep", 512),
        )
        config.update({k: v for k, v in params.items() if k not in {"initial_gadgets", "target_gadget"}})
        config.update(override_params)  # For user override
        passed = run_rl_simulation(
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
    run_all_rl_tests()
    # Example: run a single test with custom PPO params
    # run_all_rl_tests(["AP2T -> C2T"], ppo_params={"learning_rate": 0.005, "ent_coef": 0.5}, max_episodes=30)
