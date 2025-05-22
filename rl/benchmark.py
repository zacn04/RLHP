import time
import random
from collections import deque, namedtuple
from copy import deepcopy
import os
import sys
import heapq

# Add project root to path so "env" and "oop" are importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from rl.env import GadgetSimulationEnv
from oop.gadgets.gadgetdefs import (
    AntiParallel2Toggle,
    Crossing2Toggle,
    Parallel2Toggle,
    NoncrossingWireToggle,
)
from sb3_contrib.ppo_mask import MaskablePPO

# Define your tasks
TASK_CONFIGS = {
    "AP2T_to_C2T": ([AntiParallel2Toggle(), AntiParallel2Toggle()], Crossing2Toggle()),
    "C2T_to_AP2T": ([Crossing2Toggle(), Crossing2Toggle()], AntiParallel2Toggle()),
    "C2T_to_P2T": ([Crossing2Toggle(), Crossing2Toggle()], Parallel2Toggle()),
    "NWT_to_AP2T": ([NoncrossingWireToggle(), NoncrossingWireToggle()], AntiParallel2Toggle()),
}

# Helper for state signature
def get_state_sig(env):
    # Use simplified network string repr as canonical signature
    simplified = env.network.simplify()
    return repr(simplified)

# Helper heuristic: difference in string length between current and target
def heuristic(env, target):
    cur = env.network.simplify()
    target_repr = repr(target)
    return abs(len(repr(cur)) - len(target_repr))

def evaluate_rl(task_name, model, num_episodes=20, epsilon=0.1):
    """Run RL inference and record time-to-success and steps."""
    init_gadgets, target = TASK_CONFIGS[task_name]
    # reuse single env instance
    env = GadgetSimulationEnv(init_gadgets, target, max_steps=8)
    obs, _ = env.reset()

    times, steps, successes = [], [], 0
    best_trace = None
    best_time = float('inf')
    mask_cache = {}

    for ep in range(num_episodes):
        obs, _ = env.reset()
        start = time.perf_counter()
        trace = []
        for t in range(env.max_steps):
            state_key = tuple(obs) if hasattr(obs, 'tolist') else repr(obs)
            if state_key not in mask_cache:
                mask_cache[state_key] = env._build_action_mask()
            mask = mask_cache[state_key]

            # model predict
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            # epsilon-greedy
            if random.random() < epsilon:
                valid = [i for i, v in enumerate(mask) if v]
                action = random.choice(valid)

            trace.append(action)
            obs, reward, done, truncated, info = env.step(action)
            if done and reward >= 200:
                elapsed = time.perf_counter() - start
                successes += 1
                times.append(elapsed)
                steps.append(t+1)
                if elapsed < best_time:
                    best_time = elapsed
                    best_trace = trace.copy()
                break
    return {
        "method": "RL",
        "task": task_name,
        "success_rate": successes / num_episodes,
        "avg_time": sum(times)/len(times) if times else None,
        "avg_steps": sum(steps)/len(steps) if steps else None,
        "best_trace": best_trace,
        "best_time": best_time if best_time != float('inf') else None
    }

def evaluate_random(task_name, max_trials=10000):
    """Randomly sample valid sequences until success."""
    init_gadgets, target = TASK_CONFIGS[task_name]
    env = GadgetSimulationEnv(init_gadgets, target, max_steps=8)

    times, steps = [], []
    best_trace, best_time = None, float('inf')
    successes = 0

    trial = 0
    while trial < max_trials:
        trial += 1
        obs, _ = env.reset()
        start = time.perf_counter()
        trace = []
        for t in range(env.max_steps):
            mask = env._build_action_mask()
            valid = [i for i, v in enumerate(mask) if v]
            action = random.choice(valid)
            trace.append(action)
            obs, reward, done, truncated, info = env.step(action)
            if done and reward >= 200:
                elapsed = time.perf_counter() - start
                successes += 1
                times.append(elapsed)
                steps.append(t+1)
                if elapsed < best_time:
                    best_time = elapsed
                    best_trace = trace.copy()
                break
        if successes > 0:
            break

    return {
        "method": "Random",
        "task": task_name,
        "success_rate": successes / trial,
        "avg_time": sum(times)/len(times) if times else None,
        "avg_steps": sum(steps)/len(steps) if steps else None,
        "best_trace": best_trace,
        "best_time": best_time if best_time != float('inf') else None
    }

def evaluate_exhaustive(task_name, max_nodes=100000):
    """A* search over small search space, find first solution."""
    init_gadgets, target = TASK_CONFIGS[task_name]
    start_time = time.perf_counter()
    
    # Initialize A* search
    init_env = GadgetSimulationEnv(init_gadgets, target, max_steps=8)
    init_env.reset()
    
    target_repr = repr(target)
    count = 0
    heap = []
    start_sig = get_state_sig(init_env)
    heapq.heappush(heap, (heuristic(init_env, target), 0, [], init_env))
    seen = {start_sig}
    best_time = float('inf')
    best_trace = None

    while heap and count < max_nodes:
        f, depth, trace, env = heapq.heappop(heap)
        count += 1
        
        # goal check
        if repr(env.network.simplify()) == target_repr:
            elapsed = time.perf_counter() - start_time
            best_time = elapsed
            best_trace = trace
            return {
                "method": "Exhaustive",
                "task": task_name,
                "success_rate": 1.0,
                "time_to_solution": elapsed,
                "steps": depth,
                "best_trace": best_trace,
                "best_time": best_time
            }
            
        if depth >= env.max_steps:
            continue
            
        mask = env._build_action_mask()
        for a, valid in enumerate(mask):
            if not valid: continue
            new_env = deepcopy(env)
            new_env.step(a)
            sig = get_state_sig(new_env)
            if sig in seen: continue
            seen.add(sig)

            h = heuristic(new_env, target)
            heapq.heappush(heap, (depth+1 + h, depth+1, trace + [a], new_env))

    # no solution
    return {
        "method": "Exhaustive",
        "task": task_name,
        "success_rate": 0.0,
        "time_to_solution": None,
        "steps": None,
        "best_trace": None,
        "best_time": None
    }

if __name__ == "__main__":
    results = []
    for task in TASK_CONFIGS:
        model = MaskablePPO.load(f"models/mppo_{task}_20250521_latest")
        results.append(evaluate_rl(task, model))
        results.append(evaluate_random(task))
        results.append(evaluate_exhaustive(task))

    print("\nTASK | METHOD | TRACE | TIME TO FIND")
    print("-" * 60)
    for res in results:
        trace_str = str(res['best_trace']) if res['best_trace'] else "No solution"
        time_str = f"{res['best_time']:.3f}s" if res['best_time'] else "N/A"
        print(f"{res['task']} | {res['method']} | {trace_str} | {time_str}")
