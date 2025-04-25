from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import numpy as np
from oop.gadgets.gadgetdefs import (
    AntiParallel2Toggle,
    Crossing2Toggle,
    Toggle2,
    AntiParallelLocking2Toggle,
    CrossingLocking2Toggle,
    ParallelLocking2Toggle,
    Door,
    SelfClosingDoor
)
from rl.search.env import GadgetSimulationEnv
from rl.search.exhaustive.search import format_operation

class BestConstructionCallback(BaseCallback):
    """
    Callback to track and print the best construction and reward at each epoch.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.best_reward = float('-inf')
        self.best_construction = None
        self.best_operations = None

    def _on_step(self) -> bool:
        # Get the current environment
        env = self.training_env.envs[0]
        
        # Get current reward and construction
        current_reward = env.get_episode_rewards()[-1] if env.get_episode_rewards() else 0
        current_construction = env.network.simplify()
        current_operations = env.operation_history
        
        # Update best if current is better
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            self.best_construction = current_construction
            self.best_operations = current_operations.copy()
            
            # Print the new best
            print(f"\nNew best reward achieved: {self.best_reward}")
            print("Best construction so far:")
            print(self.best_construction)
            print("Operations that led to this construction:")
            for op in self.best_operations:
                print(f"  - {format_operation(op)}")
        
        return True

def plot_training_progress(rewards, title):
    """Plot training progress"""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title(f'Training Progress - {title}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig(f'training_{title.lower().replace(" ", "_")}.png')
    plt.close()

def test_ap2t_sim_c2t():
    """Test if AP2T can simulate C2T using RL."""
    print("\n=== Testing RL AP2T -> C2T simulation ===")
    
    # Create initial gadgets
    ap2t1 = AntiParallel2Toggle()
    ap2t2 = AntiParallel2Toggle()
    target = Crossing2Toggle()
    
    # Create and verify environment with increased max_steps for training
    original_env = GadgetSimulationEnv([ap2t1, ap2t2], target, max_steps=200)
    check_env(original_env)
    
    # Create vectorized environment
    env = DummyVecEnv([lambda: original_env])
    
    # Create callback
    best_callback = BestConstructionCallback()
    
    # Create and train PPO agent with increased exploration and better learning parameters
    model = PPO(
        "MultiInputPolicy", 
        env, 
        learning_rate=0.0003,  # Increased learning rate
        n_steps=512,  # Reduced for more frequent updates
        batch_size=64,  # Smaller batches for more exploration
        n_epochs=10,  # Fewer epochs to prevent overfitting
        gamma=0.95,  # Reduced discount factor
        gae_lambda=0.9,  # Reduced GAE lambda
        clip_range=lambda _: 0.4,  # Make clip_range a callable
        ent_coef=0.2,  # Higher entropy coefficient for more exploration
        verbose=1
    )
    
    # Train for more episodes with better progress tracking
    rewards = []
    best_reward = float('-inf')
    episodes_without_improvement = 0
    max_episodes = 200  # Increased to 200 episodes
    exploration_phase = 0  # Track exploration phase
    
    for episode in range(max_episodes):
        # Train for one episode
        model.learn(total_timesteps=512, callback=best_callback)  # Add callback here
        
        # Evaluate
        obs = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        while not done:
            # Progressive exploration strategy
            if episodes_without_improvement >= 10:
                if exploration_phase == 0:
                    # Phase 1: Mix random actions with policy
                    if step_count % 5 == 0:
                        action = [env.action_space.sample()]
                    else:
                        action, _states = model.predict(obs, deterministic=False)
                elif exploration_phase == 1:
                    # Phase 2: More random actions
                    if step_count % 3 == 0:
                        action = [env.action_space.sample()]
                    else:
                        action, _states = model.predict(obs, deterministic=False)
                else:
                    # Phase 3: Mostly random actions
                    if step_count % 2 == 0:
                        action = [env.action_space.sample()]
                    else:
                        action, _states = model.predict(obs, deterministic=False)
            else:
                action, _states = model.predict(obs, deterministic=False)
                
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            step_count += 1
            
            # Early stopping if reward gets too negative
            if total_reward < -1000:
                print(f"Stopping early due to very negative reward: {total_reward}")
                done = True
                break
            
            # Early stopping if we achieve a good reward
            if total_reward > 100:
                print(f"Stopping early due to good reward: {total_reward}")
                done = True
                break
            
            # Log intermediate state every 50 steps
            if step_count % 50 == 0:
                print(f"\nEpisode {episode+1}, Step {step_count}:")
                print(f"Current reward: {total_reward}")
                print(f"Current number of gadgets: {len(env.envs[0].network.subgadgets)}")
                if env.envs[0].successful_operations:
                    print("Recent operations:")
                    for op in env.envs[0].successful_operations[-3:]:  # Show last 3 operations
                        print(f"  - {format_operation(op)}")
        
        rewards.append(total_reward)
        print(f"\nTraining episode {episode+1}/{max_episodes} - Total reward: {total_reward}")
        print(f"Number of gadgets: {len(env.envs[0].network.subgadgets)}")
        print(f"Steps taken: {step_count}")
        
        # Track progress
        if total_reward > best_reward:
            best_reward = total_reward
            episodes_without_improvement = 0
            exploration_phase = 0  # Reset exploration phase on improvement
            print("New best reward achieved!")
        else:
            episodes_without_improvement += 1
            print(f"No improvement for {episodes_without_improvement} episodes")
            
            # Progressive exploration phases
            if episodes_without_improvement >= 10:
                if exploration_phase < 2:  # Only increase phase if not at max
                    exploration_phase += 1
                    print(f"Entering exploration phase {exploration_phase + 1}")
                    # Create new model with adjusted parameters
                    new_clip_range = 0.4 + (0.2 * exploration_phase)
                    new_ent_coef = 0.2 + (0.2 * exploration_phase)
                    # Create new model with updated parameters
                    new_model = PPO(
                        "MultiInputPolicy", 
                        env, 
                        learning_rate=0.0003,
                        n_steps=512,
                        batch_size=64,
                        n_epochs=10,
                        gamma=0.95,
                        gae_lambda=0.9,
                        clip_range=lambda _: new_clip_range,
                        ent_coef=new_ent_coef,
                        verbose=1
                    )
                    # Transfer parameters from old model to new model
                    new_model.set_parameters(model.get_parameters())
                    model = new_model
        
        # Less aggressive early stopping
        if episodes_without_improvement >= 20 and episode >= 50:  # More lenient early stopping
            print("Early stopping due to no improvement")
            break
    
    # Plot training progress
    plot_training_progress(rewards, "AP2T to C2T")
    
    print("\nTesting trained agent...")
    # Create a new environment with shorter max_steps for testing
    test_env = GadgetSimulationEnv([ap2t1, ap2t2], target, max_steps=50)  # Shorter for testing
    test_env = DummyVecEnv([lambda: test_env])
    
    obs = test_env.reset()
    done = False
    total_reward = 0
    steps = 0
    max_test_steps = 50  # Limit test steps
    
    while not done and steps < max_test_steps:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        total_reward += reward[0]
        steps += 1
        
        # Early stopping conditions for testing
        if total_reward < -1000:
            print(f"Stopping test due to very negative reward: {total_reward}")
            break
        if total_reward > 100:
            print(f"Stopping test due to good reward: {total_reward}")
            break
        if 'error' in info[0]:
            print(f"Error: {info[0]['error']}")
            break
    
    print(f"Final total reward: {total_reward}")
    print(f"Final number of gadgets: {len(test_env.envs[0].network.subgadgets)}")
    
    print("\nOperation sequence:")
    for op in test_env.envs[0].successful_operations:
        print(format_operation(op))
    
    # Test if the simplified network matches the target
    simplified = test_env.envs[0].network.simplify()
    print("\nProposed solution:")
    print(simplified)
    print("\nTarget gadget:")
    print(target)
    
    if simplified == target:
        print("✅ Test passed: Found valid solution")
        print("\nOperation sequence:")
        for i, op in enumerate(test_env.envs[0].operation_history, 1):
            print(f"Step {i}: {format_operation(op)}")
        return True
    else:
        print("❌ Test failed: No valid solution found")
        print("  - Simplified gadget does not match target")
        return False

def test_cl2t_sim_pl2t():
    """Test if RL can learn to simulate PL2T using two CL2Ts"""
    print("\n=== Testing RL CL2T -> PL2T simulation ===")
    
    # Initial gadgets
    cl2t1 = CrossingLocking2Toggle()
    cl2t2 = CrossingLocking2Toggle()
    target = ParallelLocking2Toggle()
    
    # Create environment with increased max_steps
    original_env = GadgetSimulationEnv(
        initial_gadgets=[cl2t1, cl2t2],
        target_gadget=target,
        max_steps=200  # Increased from 50
    )
    
    # Verify environment
    check_env(original_env)
    
    # Create vectorized environment
    env = DummyVecEnv([lambda: original_env])
    
    # Create and train agent with better parameters
    model = PPO(
        "MultiInputPolicy", 
        env, 
        learning_rate=0.0003,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.9,
        clip_range=lambda _: 0.4,  # Make clip_range a callable
        ent_coef=0.2,
        verbose=1
    )
    
    # Train for more episodes with better progress tracking
    rewards = []
    best_reward = float('-inf')
    episodes_without_improvement = 0
    max_episodes = 200  # Increased to 200 episodes
    exploration_phase = 0  # Track exploration phase
    
    for episode in range(max_episodes):
        # Train for one episode
        model.learn(total_timesteps=512)
        
        # Evaluate
        obs = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        while not done:
            # Progressive exploration strategy
            if episodes_without_improvement >= 10:
                if exploration_phase == 0:
                    # Phase 1: Mix random actions with policy
                    if step_count % 5 == 0:
                        action = [env.action_space.sample()]
                    else:
                        action, _states = model.predict(obs, deterministic=False)
                elif exploration_phase == 1:
                    # Phase 2: More random actions
                    if step_count % 3 == 0:
                        action = [env.action_space.sample()]
                    else:
                        action, _states = model.predict(obs, deterministic=False)
                else:
                    # Phase 3: Mostly random actions
                    if step_count % 2 == 0:
                        action = [env.action_space.sample()]
                    else:
                        action, _states = model.predict(obs, deterministic=False)
            else:
                action, _states = model.predict(obs, deterministic=False)
                
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            step_count += 1
            
            # Early stopping if reward gets too negative
            if total_reward < -1000:
                print(f"Stopping early due to very negative reward: {total_reward}")
                done = True
                break
            
            # Early stopping if we achieve a good reward
            if total_reward > 100:
                print(f"Stopping early due to good reward: {total_reward}")
                done = True
                break
            
            # Log intermediate state every 50 steps
            if step_count % 50 == 0:
                print(f"\nEpisode {episode+1}, Step {step_count}:")
                print(f"Current reward: {total_reward}")
                print(f"Current number of gadgets: {len(env.envs[0].network.subgadgets)}")
                if env.envs[0].successful_operations:
                    print("Recent operations:")
                    for op in env.envs[0].successful_operations[-3:]:
                        print(f"  - {format_operation(op)}")
        
        rewards.append(total_reward)
        print(f"\nTraining episode {episode+1}/{max_episodes} - Total reward: {total_reward}")
        print(f"Number of gadgets: {len(env.envs[0].network.subgadgets)}")
        print(f"Steps taken: {step_count}")
        
        # Track progress
        if total_reward > best_reward:
            best_reward = total_reward
            episodes_without_improvement = 0
            exploration_phase = 0  # Reset exploration phase on improvement
            print("New best reward achieved!")
        else:
            episodes_without_improvement += 1
            print(f"No improvement for {episodes_without_improvement} episodes")
            
            # Progressive exploration phases
            if episodes_without_improvement >= 10:
                if exploration_phase < 2:  # Only increase phase if not at max
                    exploration_phase += 1
                    print(f"Entering exploration phase {exploration_phase + 1}")
                    # Create new model with adjusted parameters
                    new_clip_range = 0.4 + (0.2 * exploration_phase)
                    new_ent_coef = 0.2 + (0.2 * exploration_phase)
                    # Create new model with updated parameters
                    new_model = PPO(
                        "MultiInputPolicy", 
                        env, 
                        learning_rate=0.0003,
                        n_steps=512,
                        batch_size=64,
                        n_epochs=10,
                        gamma=0.95,
                        gae_lambda=0.9,
                        clip_range=lambda _: new_clip_range,
                        ent_coef=new_ent_coef,
                        verbose=1
                    )
                    # Transfer parameters from old model to new model
                    new_model.set_parameters(model.get_parameters())
                    model = new_model
        
        # Less aggressive early stopping
        if episodes_without_improvement >= 20 and episode >= 50:
            print("Early stopping due to no improvement")
            break
    
    # Plot training progress
    plot_training_progress(rewards, "CL2T to PL2T")
    
    # Test the trained agent
    print("\nTesting trained agent...")
    obs = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        step_count += 1
        
        # Early stopping conditions for testing
        if total_reward < -1000:
            print(f"Stopping test due to very negative reward: {total_reward}")
            break
        if total_reward > 100:
            print(f"Stopping test due to good reward: {total_reward}")
            break
        
        # Log intermediate state every 10 steps during testing
        if step_count % 10 == 0:
            print(f"\nTest Step {step_count}:")
            print(f"Current reward: {total_reward}")
            print(f"Current number of gadgets: {len(env.envs[0].network.subgadgets)}")
            if env.envs[0].successful_operations:
                print("Recent operations:")
                for op in env.envs[0].successful_operations[-3:]:
                    print(f"  - {format_operation(op)}")
        
        if "error" in info[0]:
            print(f"Error: {info[0]['error']}")
            break
            
    print(f"Final total reward: {total_reward}")
    print(f"Final number of gadgets: {len(original_env.network.subgadgets)}")
    
    # Print operation sequence
    print("\nOperation sequence:")
    for op in original_env.successful_operations:
        print(format_operation(op))
    
    # Test if the simplified network matches the target
    simplified = original_env.network.simplify()
    print("\nProposed solution:")
    print(simplified)
    print("\nTarget gadget:")
    print(target)
    
    if simplified == target:
        print("✅ Test passed: Found valid solution")
        print("\nOperation sequence:")
        for i, op in enumerate(original_env.operation_history, 1):
            print(f"Step {i}: {format_operation(op)}")
        return True
    else:
        print("❌ Test failed: No valid solution found")
        print("  - Simplified gadget does not match target")
        return False


def test_c2t_sim_p2t():
    """Test if RL can learn to simulate Toggle2 using two Crossing2Toggles"""
    print("\n=== Testing RL C2T -> Toggle2 simulation ===")
    
    # Initial gadgets
    c2t1 = Crossing2Toggle()
    c2t2 = Crossing2Toggle()
    target = Toggle2()
    
    # Create environment
    original_env = GadgetSimulationEnv(
        initial_gadgets=[c2t1, c2t2],
        target_gadget=target,
        max_steps=200  # Increased from 50
    )
    
    # Verify environment
    check_env(original_env)
    
    # Create vectorized environment
    env = DummyVecEnv([lambda: original_env])
    
    # Create and train agent with better parameters
    model = PPO(
        "MultiInputPolicy", 
        env, 
        learning_rate=0.0003,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.9,
        clip_range=lambda _: 0.4,  # Make clip_range a callable
        ent_coef=0.2,
        verbose=1
    )
    
    # Train for more episodes with better progress tracking
    rewards = []
    best_reward = float('-inf')
    episodes_without_improvement = 0
    max_episodes = 200  # Increased to 200 episodes
    exploration_phase = 0  # Track exploration phase
    
    for episode in range(max_episodes):
        # Train for one episode
        model.learn(total_timesteps=512)
        
        # Evaluate
        obs = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        while not done:
            # Progressive exploration strategy
            if episodes_without_improvement >= 10:
                if exploration_phase == 0:
                    # Phase 1: Mix random actions with policy
                    if step_count % 5 == 0:
                        action = [env.action_space.sample()]
                    else:
                        action, _states = model.predict(obs, deterministic=False)
                elif exploration_phase == 1:
                    # Phase 2: More random actions
                    if step_count % 3 == 0:
                        action = [env.action_space.sample()]
                    else:
                        action, _states = model.predict(obs, deterministic=False)
                else:
                    # Phase 3: Mostly random actions
                    if step_count % 2 == 0:
                        action = [env.action_space.sample()]
                    else:
                        action, _states = model.predict(obs, deterministic=False)
            else:
                action, _states = model.predict(obs, deterministic=False)
                
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            step_count += 1
            
            # Early stopping if reward gets too negative
            if total_reward < -1000:
                print(f"Stopping early due to very negative reward: {total_reward}")
                done = True
                break
            
            # Early stopping if we achieve a good reward
            if total_reward > 100:
                print(f"Stopping early due to good reward: {total_reward}")
                done = True
                break
            
            # Log intermediate state every 50 steps
            if step_count % 50 == 0:
                print(f"\nEpisode {episode+1}, Step {step_count}:")
                print(f"Current reward: {total_reward}")
                print(f"Current number of gadgets: {len(env.envs[0].network.subgadgets)}")
                if env.envs[0].successful_operations:
                    print("Recent operations:")
                    for op in env.envs[0].successful_operations[-3:]:
                        print(f"  - {format_operation(op)}")
        
        rewards.append(total_reward)
        print(f"\nTraining episode {episode+1}/{max_episodes} - Total reward: {total_reward}")
        print(f"Number of gadgets: {len(env.envs[0].network.subgadgets)}")
        print(f"Steps taken: {step_count}")
        
        # Track progress
        if total_reward > best_reward:
            best_reward = total_reward
            episodes_without_improvement = 0
            exploration_phase = 0  # Reset exploration phase on improvement
            print("New best reward achieved!")
        else:
            episodes_without_improvement += 1
            print(f"No improvement for {episodes_without_improvement} episodes")
            
            # Progressive exploration phases
            if episodes_without_improvement >= 10:
                if exploration_phase < 2:  # Only increase phase if not at max
                    exploration_phase += 1
                    print(f"Entering exploration phase {exploration_phase + 1}")
                    # Create new model with adjusted parameters
                    new_clip_range = 0.4 + (0.2 * exploration_phase)
                    new_ent_coef = 0.2 + (0.2 * exploration_phase)
                    # Create new model with updated parameters
                    new_model = PPO(
                        "MultiInputPolicy", 
                        env, 
                        learning_rate=0.0003,
                        n_steps=512,
                        batch_size=64,
                        n_epochs=10,
                        gamma=0.95,
                        gae_lambda=0.9,
                        clip_range=lambda _: new_clip_range,
                        ent_coef=new_ent_coef,
                        verbose=1
                    )
                    # Transfer parameters from old model to new model
                    new_model.set_parameters(model.get_parameters())
                    model = new_model
        
        # Less aggressive early stopping
        if episodes_without_improvement >= 20 and episode >= 50:
            print("Early stopping due to no improvement")
            break
    
    # Plot training progress
    plot_training_progress(rewards, "C2T to Toggle2")
    
    # Test the trained agent
    print("\nTesting trained agent...")
    obs = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        step_count += 1
        
        # Early stopping conditions for testing
        if total_reward < -1000:
            print(f"Stopping test due to very negative reward: {total_reward}")
            break
        if total_reward > 100:
            print(f"Stopping test due to good reward: {total_reward}")
            break
        
        # Log intermediate state every 10 steps during testing
        if step_count % 10 == 0:
            print(f"\nTest Step {step_count}:")
            print(f"Current reward: {total_reward}")
            print(f"Current number of gadgets: {len(env.envs[0].network.subgadgets)}")
            if env.envs[0].successful_operations:
                print("Recent operations:")
                for op in env.envs[0].successful_operations[-3:]:
                    print(f"  - {format_operation(op)}")
        
        if "error" in info[0]:
            print(f"Error: {info[0]['error']}")
            break
            
    print(f"Final total reward: {total_reward}")
    print(f"Final number of gadgets: {len(original_env.network.subgadgets)}")
    
    # Print operation sequence
    print("\nOperation sequence:")
    for op in original_env.successful_operations:
        print(format_operation(op))
    
    # Test if the simplified network matches the target
    simplified = original_env.network.simplify()
    print("\nProposed solution:")
    print(simplified)
    print("\nTarget gadget:")
    print(target)
    
    if simplified == target:
        print("✅ Test passed: Found valid solution")
        print("\nOperation sequence:")
        for i, op in enumerate(original_env.operation_history, 1):
            print(f"Step {i}: {format_operation(op)}")
        return True
    else:
        print("❌ Test failed: No valid solution found")
        print("  - Simplified gadget does not match target")
        return False

def run_all_tests():
    """Run all RL tests"""
    print("\n=== Running All RL Tests ===")
    tests = [
        ("AP2T -> C2T", test_ap2t_sim_c2t),
        ("CL2T -> PL2T", test_cl2t_sim_pl2t),
        ("C2T -> Toggle2", test_c2t_sim_p2t)
    ]
    
    results = []
    for name, test in tests:
        print(f"\nStarting test: {name}")
        result = test()
        results.append(result)
        print(f"Test {name}: {'✅ PASSED' if result else '❌ FAILED'}")
    
    print("\n=== Test Summary ===")
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(tests) - sum(results)}")
    
    return all(results)

if __name__ == "__main__":
    run_all_tests() 