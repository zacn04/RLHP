from typing import List, Tuple, Optional, Dict
import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces
from oop.gadgets.gadgetdefs import *
from oop.gadgets.gadgetlike import Gadget, GadgetNetwork
from oop.dfa.hopcroft import list_hopcroft_minimisation, list_normalisation
from .exhaustive.search import get_possible_operations, apply_operation, network_state_hash

class GadgetSimulationEnv(Env):
    def __init__(self, initial_gadgets: List[Gadget], target_gadget: Gadget, max_steps: int = 100):
        super().__init__()
        
        self.initial_gadgets = initial_gadgets
        self.target_gadget = target_gadget
        self.max_steps = max_steps
        max_gadgets = len(initial_gadgets) * 10
        self.max_gadgets = max_gadgets
        # Initialize state
        self.network = GadgetNetwork()
        for gadget in initial_gadgets:
            self.network += gadget
            
        # Action space is now only operations (combine, connect)
        self.action_space = spaces.Discrete(1)  # Placeholder, will be updated in reset()
        
        # Observation space: 
        # - Number of gadgets in network
        # - and whats happenig with each gadget
        ''' 'gadget_types': spaces.Box(low=0, high=np.inf, shape=(len(initial_gadgets),), dtype=np.int32),
            'gadget_states': spaces.Box(low=0, high=np.inf, shape=(len(initial_gadgets),), dtype=np.int32),
            'connections': spaces.Box(low=0, high=np.inf, shape=(len(initial_gadgets) * len(initial_gadgets),), dtype=np.int32)  # Flattened '''
        self.observation_space = spaces.Dict({
            'gadgets': spaces.Box(low=-1, high=100, shape=(max_gadgets * 5, ), dtype=np.int32),
            'num_gadgets': spaces.Box(low=0, high=max_gadgets, shape=(1,), dtype=np.int32),
            })
            
        
        
        self.steps = 0
        self.visited_states = set()
        self.operation_history = []  # Track sequence of operations in current episode
        self.successful_operations = []  # Track operations that led to a solution
        self.best_partial_solution = {
            'operations': [],
            'similarity': 0.0,
            'network': None
        }
        
    def _get_observation(self):
        """
        Construct an observation for RL based on the current list of gadgets in the network.
        Each gadget is encoded as:
            [type_id, state, is_composite, parent1, parent2]
        All lists are padded to a fixed length (max_gadgets).
        """
        # ---- CONFIGURATION ----
        max_gadgets = len(self.initial_gadgets) * 10  # Max gadgets ever present (can tune)
        obs_gadgets = []
        # Map each gadget class to an integer for RL; ensure this is a global/static mapping
        GADGET_TYPE_MAP = {
            # Fill in all your relevant types
            AntiParallel2Toggle: 0,
            Crossing2Toggle: 1,
            Toggle2: 2,
            AntiParallelLocking2Toggle: 3,
            CrossingLocking2Toggle: 4,
            ParallelLocking2Toggle: 5,
            Door: 6,
            SelfClosingDoor: 7,
            # Add more as needed
        }

        # ---- ENCODE EACH GADGET ----
        for idx, g in enumerate(self.network.subgadgets):
            # Type as integer
            type_id = GADGET_TYPE_MAP.get(type(g), -1)  # Unknown type = -1
            # Current state (int, or first elem if tuple)
            state = g.getCurrentState() if hasattr(g, 'getCurrentState') else -1
            # Is composite: 1 if has parent_indices, else 0
            if hasattr(g, 'parent_indices'):
                is_composite = 1
                parents = list(g.parent_indices)
            else:
                is_composite = 0
                parents = [-1, -1]
            if len(parents) < 2:
                parents += [-1] * (2 - len(parents))
            # Optionally, include idx to help agent with self-referencing (may help in advanced RL)
            obs_gadgets.append([type_id, state, is_composite] + parents)

        # ---- PAD TO FIXED LENGTH ----
        while len(obs_gadgets) < max_gadgets:
            obs_gadgets.append([-1, -1, 0, -1, -1])  # Padding for unused slots

        obs_gadgets = np.array(obs_gadgets, dtype=np.int32).flatten()  # Shape: (max_gadgets, 5)

        # ---- (Optional) ADD EXTRA INFO ----
        num_gadgets = len(self.network.subgadgets)
        obs = {
            'gadgets': obs_gadgets,                # Main state, shape (max_gadgets, 5)
            'num_gadgets': np.array([num_gadgets], dtype=np.int32),
            
        }
        return obs

    
    def _get_reward(self) -> float:
        """Calculate reward based on current state and partial progress"""
        reward = 0.0

        # Check if we've found the target gadget
        simplified = self.network.simplify()
        
        # Calculate behavioral similarity
        similarity_score = self._calculate_similarity(simplified, self.target_gadget)
        
        # Track best partial solution
        if similarity_score > self.best_partial_solution['similarity']:
            self.best_partial_solution = {
                'operations': self.operation_history.copy(),
                'similarity': similarity_score,
                'network': simplified
            }
        
        # Base reward on similarity (binary: 0 or 1)
        reward += similarity_score * 10.0  # Reward for behavioral equivalence
        
        # Small reward for attempting to combine gadgets
        if self.operation_history and self.operation_history[-1][0] == "COMBINE":
            reward += 0.5  # Small reward for attempting to combine
        
        # Small penalty for each step to encourage finding solutions quickly
        reward -= 0.1
        
        return reward
    
    def _calculate_similarity(self, gadget1: Gadget, gadget2: Gadget) -> float:
        """Calculate similarity between two gadgets based on behavioral equivalence using Hopcroft minimization."""
        try:
            # Get reachable parts of both gadgets
            reachable_self = gadget1._get_reachable_gadget()
            reachable_other = gadget2._get_reachable_gadget()
            
            # Prepare DFAs for minimization
            dfa1 = (
                reachable_self.getStates(), 
                reachable_self.getLocations(), 
                reachable_self.getTransitions(), 
                reachable_self.current_state, 
                reachable_self.getStates()[1:] if len(reachable_self.getStates()) > 1 else []
            )
            
            dfa2 = (
                reachable_other.getStates(), 
                reachable_other.getLocations(), 
                reachable_other.getTransitions(), 
                reachable_other.current_state, 
                reachable_other.getStates()[1:] if len(reachable_other.getStates()) > 1 else []
            )
            
            # Use Hopcroft minimization
            min_dfa1 = list_hopcroft_minimisation(*dfa1)
            min_dfa2 = list_hopcroft_minimisation(*dfa2)
            
            # Normalize location numbering for comparison
            norm_dfa1 = list_normalisation(min_dfa1)
            norm_dfa2 = list_normalisation(min_dfa2)
            
            # Check isomorphism between the minimized DFAs
            if gadget1._are_dfas_isomorphic(norm_dfa1, norm_dfa2):
                return 1.0  # Full similarity if behaviorally equivalent
            return 0.0  # No similarity if not behaviorally equivalent
            
        except Exception as e:
            print(f"Error in similarity calculation: {str(e)}")
            return 0.0
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.network = GadgetNetwork()
        for gadget in self.initial_gadgets:
            self.network += gadget
            
        self.steps = 0
        self.visited_states = set()
        self.operation_history = []
        
        # Update action space to only include operations
        possible_ops = get_possible_operations(self.network)
        self.action_space = spaces.Discrete(len(possible_ops))
        
        return self._get_observation(), {}
    
    def step(self, action_idx: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.steps += 1
        ops = get_possible_operations(self.network)
        obs = self._get_observation()

        # Invalid index check
        if action_idx < 0 or action_idx >= len(ops):
            return obs, -1.0, True, False, {'error': 'Action out of range'}

        op = ops[action_idx]

        # 1) STOP action
        if op[0] == "STOP":
            final = self.network.simplify()
            success = (final == self.target_gadget)
            reward = 10.0 if success else -1.0
            # Natural termination—no truncation
            return obs, reward, True, False, {}

        # 2) Otherwise, COMBINE or CONNECT
        #   Attempt to apply the op on a fresh copy
        new_net = apply_operation(self.network, op)
        if new_net is None:
            # illegal operation = logical failure
            return obs, -1.0, True, False, {'error': 'Invalid operation'}

        # 3) Duplicate‐state check (non-terminal)
        h = network_state_hash(new_net)
        if h in self.visited_states:
            return obs, -0.5, False, False, {'info': 'Duplicate state'}

        # 4) Accept the new state
        self.visited_states.add(h)
        self.network = new_net
        self.operation_history.append(op)

        # 5) Compute reward and check for automatic solve
        reward = self._get_reward()
        if reward >= 0.99:
            # auto-stop on success
            return self._get_observation(), reward + 2.0, True, False, {}

        # 6) Step budget check → truncation
        if self.steps >= self.max_steps:
            return self._get_observation(), reward, False, True, {}

        # 7) Otherwise, continue
        return self._get_observation(), reward, False, False, {}
