from typing import List, Tuple, Optional, Dict
import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces
from oop.gadgets.gadgetlike import Gadget, GadgetNetwork
from .exhaustive.search import get_possible_operations, apply_operation, network_state_hash

class GadgetSimulationEnv(Env):
    def __init__(self, initial_gadgets: List[Gadget], target_gadget: Gadget, max_steps: int = 100):
        super().__init__()
        
        self.initial_gadgets = initial_gadgets
        self.target_gadget = target_gadget
        self.max_steps = max_steps
        
        # Initialize state
        self.network = GadgetNetwork()
        for gadget in initial_gadgets:
            self.network += gadget
            
        # Action space is now:
        # - Operations (combine, connect, etc.)
        # - State changes for each gadget
        self.action_space = spaces.Discrete(1)  # Placeholder, will be updated in reset()
        
        # Observation space: 
        # - Number of gadgets in network
        # - For each gadget: type, state, connections
        # - Target gadget info
        self.observation_space = spaces.Dict({
            'num_gadgets': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
            'gadget_types': spaces.Box(low=0, high=np.inf, shape=(len(initial_gadgets),), dtype=np.int32),
            'gadget_states': spaces.Box(low=0, high=np.inf, shape=(len(initial_gadgets),), dtype=np.int32),
            'connections': spaces.Box(low=0, high=np.inf, shape=(len(initial_gadgets) * len(initial_gadgets),), dtype=np.int32)  # Flattened
        })
        
        self.steps = 0
        self.visited_states = set()
        self.operation_history = []  # Track sequence of operations in current episode
        self.successful_operations = []  # Track operations that led to a solution
        
    def _get_observation(self) -> Dict:
        """Convert current network state to observation space"""
        num_gadgets = len(self.network.subgadgets)
        
        # Create arrays with fixed size based on initial gadgets
        gadget_types = np.zeros(len(self.initial_gadgets), dtype=np.int32)
        gadget_states = np.zeros(len(self.initial_gadgets), dtype=np.int32)
        
        # Fill arrays with current gadget info
        for i, gadget in enumerate(self.network.subgadgets):
            if i < len(self.initial_gadgets):  # Only fill up to initial size
                gadget_types[i] = hash(type(gadget))
                # Convert tuple state to integer by taking first element
                state = gadget.getCurrentState()
                if isinstance(state, tuple):
                    gadget_states[i] = state[0]
                else:
                    gadget_states[i] = int(state)
        
        # Create connection matrix and flatten it
        connections = np.zeros((len(self.initial_gadgets), len(self.initial_gadgets)), dtype=np.int32)
        # TODO: Fill in actual connections
        connections = connections.flatten()  # Flatten to 1D array
        
        # Convert to tensors for the policy
        return {
            'num_gadgets': np.array([num_gadgets], dtype=np.int32),
            'gadget_types': np.array(gadget_types, dtype=np.int32),
            'gadget_states': np.array(gadget_states, dtype=np.int32),
            'connections': np.array(connections, dtype=np.int32)
        }
    
    def _get_reward(self) -> float:
        """Calculate reward based on current state and partial progress"""
        reward = 0.0
        
        # No base reward for valid actions - only reward progress
        # reward += 0.1  # Removed base reward
        
        # Stronger penalty for extra gadgets
        extra_gadgets = len(self.network.subgadgets) - len(self.initial_gadgets)
        reward -= 5.0 * extra_gadgets  # Increased penalty
        
        # Check if we've found the target gadget
        simplified = self.network.simplify()
        
        # Calculate DFA similarity score with stricter matching
        similarity_score = self._calculate_similarity(simplified, self.target_gadget)
        
        # Only give similarity reward if we're making progress
        if similarity_score > 0.8:  # Only reward high similarity
            reward += similarity_score * 2.0  # Reduced weight of similarity
            
        # Additional reward for reducing number of gadgets
        if len(self.network.subgadgets) < len(self.initial_gadgets):
            reward += 2.0  # Increased reward for combining gadgets
            
        # Large reward for finding exact solution
        if simplified == self.target_gadget and len(self.network.subgadgets) == 1:
            if len(self.operation_history) > 0:  # Must have taken at least one action
                # Store successful operations
                self.successful_operations = self.operation_history.copy()
                reward = 100.0  # Much higher reward for solution
        
        # Cap negative rewards at -20 (more negative to discourage bad behavior)
        reward = max(-20.0, reward)
        
        return reward
    
    def _calculate_similarity(self, gadget1: Gadget, gadget2: Gadget) -> float:
        """Calculate similarity between two gadgets based on their DFA properties"""
        score = 0.0
        
        # Compare number of states - must be exact match
        states1 = len(gadget1.getStates())
        states2 = len(gadget2.getStates())
        if states1 == states2:
            score += 0.4  # Only reward exact match
        else:
            return 0.0  # Return 0 if states don't match
            
        # Compare transitions - must be exact match
        trans1 = gadget1.getTransitions()
        trans2 = gadget2.getTransitions()
        if set(trans1) == set(trans2):
            score += 0.4  # Only reward exact match
        else:
            return 0.0  # Return 0 if transitions don't match
            
        # Compare current states - must be exact match
        if gadget1.getCurrentState() == gadget2.getCurrentState():
            score += 0.2  # Only reward exact match
        else:
            return 0.0  # Return 0 if current states don't match
            
        return score  # No normalization needed since we only reward exact matches
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.network = GadgetNetwork()
        for gadget in self.initial_gadgets:
            self.network += gadget
            
        self.steps = 0
        self.visited_states = set()
        self.operation_history = []
        
        # Update action space to include both operations and state changes
        possible_ops = get_possible_operations(self.network)
        total_states = sum(len(gadget.getStates()) for gadget in self.network.subgadgets)
        self.action_space = spaces.Discrete(len(possible_ops) + total_states)
        
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        self.steps += 1
        
        # Get possible operations and state changes
        possible_ops = get_possible_operations(self.network)
        num_ops = len(possible_ops)
        
        if action < num_ops:
            # Perform operation
            operation = possible_ops[action]
            self.operation_history.append(operation)
            new_network = apply_operation(self.network, operation)
        else:
            # Change state of a gadget
            state_action = action - num_ops
            total_states = 0
            gadget_idx = None
            state_idx = None
            
            # Find the correct gadget and state
            for i, gadget in enumerate(self.network.subgadgets):
                num_states = len(gadget.getStates())
                if state_action < total_states + num_states:
                    gadget_idx = i
                    state_idx = state_action - total_states
                    break
                total_states += num_states
            
            if gadget_idx is not None and state_idx is not None:
                gadget = self.network.subgadgets[gadget_idx]
                states = list(gadget.getStates())  # Convert to list to ensure ordered access
                if state_idx < len(states):
                    try:
                        gadget.setCurrentState(states[state_idx])
                        new_network = self.network
                    except ValueError:
                        return self._get_observation(), -1.0, True, False, {'error': 'Invalid state value'}
                else:
                    return self._get_observation(), -1.0, True, False, {'error': 'Invalid state index'}
            else:
                return self._get_observation(), -1.0, True, False, {'error': 'Invalid gadget index'}
            
        if new_network is None:
            return self._get_observation(), -1.0, True, False, {'error': 'Invalid operation'}
            
        # Check for duplicate states
        state_hash = network_state_hash(new_network)
        if state_hash in self.visited_states:
            return self._get_observation(), -0.5, False, False, {'info': 'Duplicate state'}
            
        self.visited_states.add(state_hash)
        self.network = new_network
        
        # Calculate reward
        reward = self._get_reward()
        
        # Check if done
        done = (self.steps >= self.max_steps or 
                (reward >= 100.0 and len(self.network.subgadgets) == 1) or  # Found exact solution
                reward <= -100.0)  # Terminate if reward is too negative
        
        return self._get_observation(), reward, done, False, {} 