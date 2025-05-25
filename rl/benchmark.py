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

from train import decode_action

def is_planar_connection(env, gadget_idx, port1, port2):
    """Check if connection is planar and matches original port labels."""
    # Get the gadget and its actual port labels
    g = env.network.subgadgets[gadget_idx]
    ports = g.getLocations()
    
    # Get the actual port labels
    try:
        port1_label = ports[port1]
        port2_label = ports[port2]
    except IndexError:
        return False
    
    # Valid connections are:
    # 1. Adjacent ports (X+1, X+2)
    # 2. Opposite ports (X+3, X+4)
    # NOT X+5 as it's on the same gadget
    valid_ports = [
        (port1_label + 1) % 8,  # adjacent 1
        (port1_label + 2) % 8,  # adjacent 2
        (port1_label + 3) % 8,  # opposite 1
        (port1_label + 4) % 8,  # opposite 2
    ]
    
    # Check if port2 is in valid positions AND matches the expected label
    return port2_label in valid_ports

def build_action_mask(env, enforce_planar=False):
    """Build action mask, optionally enforcing planar connections."""
    mask = env._build_action_mask()
    
    if not enforce_planar:
        return mask
        
    # For each CONNECT action, check if it's planar
    for i, valid in enumerate(mask):
        if not valid:
            continue
            
        action = decode_action(i, env)
        if action[0] == 'CONNECT':
            gadget_idx, port1, port2 = action[1:]
            if not is_planar_connection(env, gadget_idx, port1, port2):
                mask[i] = False
                
    return mask

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

def evaluate_rl(task_name, model, num_episodes=20, epsilon=0.1, enforce_planar=False):
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
                mask_cache[state_key] = build_action_mask(env, enforce_planar)
            mask = mask_cache[state_key]

            # model predict
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            # epsilon-greedy
            if random.random() < epsilon:
                valid = [i for i, v in enumerate(mask) if v]
                action = random.choice(valid)

            # Get the actual operation with true port labels
            op = env.op_from_action(action)
            trace.append(op)
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

def evaluate_random(task_name, max_trials=10000, enforce_planar=False):
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
            mask = build_action_mask(env, enforce_planar)
            valid = [i for i, v in enumerate(mask) if v]
            action = random.choice(valid)
            
            # Get the actual operation with true port labels
            op = env.op_from_action(action)
            trace.append(op)
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

def evaluate_exhaustive(task_name, max_nodes=10000, time_limit=3600, enforce_planar=False):  # 60 minute limit per task
    """BFS over small search space with aggressive pruning."""
    init_gadgets, target = TASK_CONFIGS[task_name]
    start_time = time.perf_counter()
    
    # Initialize BFS
    init_env = GadgetSimulationEnv(init_gadgets, target, max_steps=8)
    init_env.reset()
    
    target_repr = repr(target)
    frontier = deque([(0, [], init_env)])  # (depth, trace, env)
    seen = {get_state_sig(init_env)}
    best_time = float('inf')
    best_trace = None
    count = 0

    # Early stopping if we find any solution
    found_solution = False

    while frontier and count < max_nodes and not found_solution:
        # Check time limit
        if time.perf_counter() - start_time > time_limit:
            break
            
        depth, trace, env = frontier.popleft()
        count += 1
        
        # goal check
        if repr(env.network.simplify()) == target_repr:
            elapsed = time.perf_counter() - start_time
            best_time = elapsed
            best_trace = trace
            found_solution = True
            break
            
        if depth >= env.max_steps:
            continue
            
        # Get valid actions and sort by potential progress
        mask = build_action_mask(env, enforce_planar)
        valid_actions = [(a, env) for a, valid in enumerate(mask) if valid]
        
        # Try actions in order, but limit branching
        for a, _ in valid_actions[:4]:  # Only try first 4 valid actions
            new_env = deepcopy(env)
            new_env.step(a)
            sig = get_state_sig(new_env)
            if sig in seen: continue
            seen.add(sig)
            # Get the actual operation with true port labels
            op = new_env.op_from_action(a)
            frontier.append((depth+1, trace + [op], new_env))

    if found_solution:
        return {
            "method": "Exhaustive",
            "task": task_name,
            "success_rate": 1.0,
            "time_to_solution": best_time,
            "steps": len(best_trace),
            "best_trace": best_trace,
            "best_time": best_time
        }
    else:
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
    total_start = time.perf_counter()
    
    # Store results by task and method for analysis
    task_results = {task: {"RL": [], "Random": [], "Exhaustive": []} for task in TASK_CONFIGS}
    
    # Toggle planar connections here
    ENFORCE_PLANAR = True
    
    for task in TASK_CONFIGS:
        # Check total time limit
        if time.perf_counter() - total_start > 3600:  # 1 hour total limit
            print("\nReached 1 hour time limit, stopping...")
            break
            
        print(f"\nEvaluating task: {task}")
        model = MaskablePPO.load(f"models/mppo_{task}_latest")
        results.append(evaluate_rl(task, model, enforce_planar=ENFORCE_PLANAR))
        results.append(evaluate_random(task, enforce_planar=ENFORCE_PLANAR))
        results.append(evaluate_exhaustive(task, enforce_planar=ENFORCE_PLANAR))
        
        # Store results for analysis
        for res in results[-3:]:
            task_results[task][res["method"]].append(res)

    # Print detailed results
    print("\n" + "="*80)
    print("DETAILED BENCHMARK RESULTS")
    print("="*80)
    
    # Print per-task results
    for task in TASK_CONFIGS:
        print(f"\n{task}:")
        print("-"*40)
        for method in ["RL", "Random", "Exhaustive"]:
            res = task_results[task][method][0] if task_results[task][method] else None
            if res:
                print(f"\n{method}:")
                print(f"  Success Rate: {res['success_rate']*100:.1f}%")
                print(f"  Best Time: {res['best_time']:.3f}s" if res['best_time'] else "  Best Time: N/A")
                print(f"  Avg. Steps: {res['avg_steps']:.1f}" if res.get('avg_steps') else "  Steps: N/A")
                print(f"  Trace: {res['best_trace']}" if res['best_trace'] else "  Trace: No solution")
            else:
                print(f"\n{method}: No results available")

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Calculate averages across all tasks
    method_stats = {
        "RL": {"success": 0, "time": [], "steps": []},
        "Random": {"success": 0, "time": [], "steps": []},
        "Exhaustive": {"success": 0, "time": [], "steps": []}
    }
    
    for task in TASK_CONFIGS:
        for method in ["RL", "Random", "Exhaustive"]:
            res = task_results[task][method][0] if task_results[task][method] else None
            if res:
                method_stats[method]["success"] += res["success_rate"]
                if res["best_time"]:
                    method_stats[method]["time"].append(res["best_time"])
                if res.get("avg_steps"):
                    method_stats[method]["steps"].append(res["avg_steps"])
    
    # Print summary
    print("\nMethod Comparison:")
    print("-"*60)
    print(f"{'Method':<10} {'Success Rate':<15} {'Avg Time (s)':<15} {'Avg Steps':<15}")
    print("-"*60)
    
    for method in ["RL", "Random", "Exhaustive"]:
        stats = method_stats[method]
        success_rate = stats["success"] / len(TASK_CONFIGS) * 100
        avg_time = sum(stats["time"])/len(stats["time"]) if stats["time"] else None
        avg_steps = sum(stats["steps"])/len(stats["steps"]) if stats["steps"] else None
        
        print(f"{method:<10} {success_rate:>6.1f}%        {avg_time:>8.3f}s        {avg_steps:>8.1f}" if avg_time and avg_steps else 
              f"{method:<10} {success_rate:>6.1f}%        {'N/A':>8}        {'N/A':>8}")
