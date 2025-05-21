from rl.expert import EXPERT_SOLUTIONS
from rl.env import GadgetSimulationEnv
from oop.gadgets.gadgetdefs import *  # import your gadgets

TASK_CONFIGS = {
    "AP2T_to_C2T": ([AntiParallel2Toggle(), AntiParallel2Toggle()], Crossing2Toggle()),
    "C2T_to_AP2T": ([Crossing2Toggle(), Crossing2Toggle()], AntiParallel2Toggle()),
    "C2T_to_P2T": ([Crossing2Toggle(), Crossing2Toggle()], Parallel2Toggle()),
    "NWT_to_AP2T": ([NoncrossingWireToggle(), NoncrossingWireToggle()], AntiParallel2Toggle()),
}

for name, (init, target) in TASK_CONFIGS.items():
    print(f"\n=== {name} ===")
    env = GadgetSimulationEnv(initial_gadgets=init, target_gadget=target, max_steps=8)
    obs = env.reset()
    success = False
    for i, op in enumerate(EXPERT_SOLUTIONS[name][0]):
        action = env.action_from_op(op)
        obs, reward, done, trunc, info = env.step(action)
        print(f"Step {i}: {op} → reward={reward}, done={done}")
        if 'error' in info:
            print(f"⚠️ Error: {info['error']}")
            break
        if done:
            success = reward >= 200
            break
    print("✅ Success!" if success else "❌ Failure.")
