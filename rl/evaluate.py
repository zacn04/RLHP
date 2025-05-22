#!/usr/bin/env python3
import os, sys, numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from env import GadgetSimulationEnv
from oop.gadgets.gadgetdefs import *
from train import decode_action, TASK_CONFIGS

def mask_fn(env):
    mask = env._build_action_mask()
    mask[-1] = 1        
    return mask

def evaluate_task(model, task_config, num_episodes=5):
    """Evaluate model on a specific task configuration."""
    env = GadgetSimulationEnv(
        initial_gadgets=task_config.initial_gadgets,
        target_gadget=task_config.target_gadget,
        max_steps=8,
    )
    env = ActionMasker(env, mask_fn)
    
    total_rewards = []
    success_count = 0
    episode_actions = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        actions = []
        
        while not done:
            mask = obs["action_mask"]
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
            actions.append(decode_action(action, env.env))
            
            if done and reward > 0:
                success_count += 1
                
        total_rewards.append(episode_reward)
        episode_actions.append(actions)
    
    return {
        'task_name': task_config.name,
        'avg_reward': np.mean(total_rewards),
        'success_rate': success_count / num_episodes,
        'rewards': total_rewards,
        'episode_actions': episode_actions
    }

def main():
    model_path = "models/mppo_NWT_to_AP2T_20250521_200610"   
    model = MaskablePPO.load(model_path)
    
    print("\nEvaluating model across all tasks:")
    print("-" * 50)
    
    results = {}
    for task_name, task_config in TASK_CONFIGS.items():
        print(f"\nEvaluating task: {task_name}")
        result = evaluate_task(model, task_config)
        results[task_name] = result
        
        print(f"Average reward: {result['avg_reward']:.2f}")
        print(f"Success rate: {result['success_rate']:.2%}")
        print(f"Individual rewards: {[f'{r:.2f}' for r in result['rewards']]}")
        print("\nAction sequences:")
        for i, actions in enumerate(result['episode_actions']):
            print(f"Episode {i+1}: {' -> '.join(actions)}")
    
    print("\nSummary:")
    print("-" * 50)
    for task_name, result in results.items():
        print(f"{task_name}:")
        print(f"  Success rate: {result['success_rate']:.2%}")
        print(f"  Average reward: {result['avg_reward']:.2f}")

if __name__ == "__main__":
    main()
