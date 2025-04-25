from enum import Enum
from itertools import product, permutations
from typing import List, Set, Dict, Tuple, Optional
from collections import deque
from copy import deepcopy
import random

from oop.gadgets.gadgetdefs import AntiParallel2Toggle, Crossing2Toggle, Door, SelfClosingDoor
from oop.gadgets.gadgetlike import Gadget, GadgetNetwork

class SearchStrategy(Enum):
    DFS = "dfs"
    BFS = "bfs"
    RANDOM = "random"

def print_gadget_definition(gadget: Gadget) -> str:
    """Print a gadget's definition in a readable format."""
    output = []
    output.append(f"Gadget: {gadget.name}")
    output.append(f"Locations: {sorted(gadget.getLocations())}")
    output.append(f"States: {sorted(gadget.getStates())}")
    output.append("Transitions:")
    
    for state in sorted(gadget.transitions.keys()):
        output.append(f"  State {state}:")
        sorted_transitions = sorted(gadget.transitions[state], 
                                 key=lambda x: (x[0], x[1]))
        for loc1, loc2, next_state in sorted_transitions:
            output.append(f"    ({loc1} → {loc2}) ⟹ State {next_state}")
    
    return "\n".join(output)

def format_operation(op: Tuple) -> str:
    """Format an operation into a human-readable string."""
    if op[0] == "COMBINE":
        _, g1_idx, g2_idx, rotation, splice_idx = op
        return f"Combine gadget {g1_idx} with gadget {g2_idx} (rotation: {rotation}, splice at: {splice_idx})"
    else:  # CONNECT
        _, gadget_idx, loc1, loc2 = op
        return f"Connect locations {loc1} and {loc2} in gadget {gadget_idx}"

def get_possible_operations(network: GadgetNetwork) -> List[Tuple]:
    """Get all valid operations for the current network."""
    operations = []
    
    # COMBINE operations
    if len(network.subgadgets) > 1:
        for i, g1 in enumerate(network.subgadgets):
            for j, g2 in enumerate(network.subgadgets[i+1:], i+1):
                for rotation in range(len(g2.getLocations())):
                    for splice_idx in range(len(g1.getLocations())):
                        operations.append(("COMBINE", i, j, rotation, splice_idx))
    
    # CONNECT operations
    for i, gadget in enumerate(network.subgadgets):
        locs = gadget.getLocations()
        for loc1, loc2 in permutations(locs, 2):
            operations.append(("CONNECT", i, loc1, loc2))
    
    return operations

def has_duplicate_locations(gadget: Gadget) -> bool:
    """Check if a gadget has duplicate locations."""
    locations = gadget.getLocations()
    return len(locations) != len(set(locations))

def validate_gadget(gadget: Gadget, operation: Tuple) -> Optional[str]:
    """Validate a gadget's locations and return error message if invalid."""
    if has_duplicate_locations(gadget):
        locs = gadget.getLocations()
        duplicates = [x for x in locs if locs.count(x) > 1]
        return f"Duplicate locations {duplicates} after {operation}"
    return None

def apply_operation(network: GadgetNetwork, operation: Tuple) -> Optional[GadgetNetwork]:
    """Apply an operation to the network with validation."""
    new_network = GadgetNetwork()
    new_network.subgadgets = deepcopy(network.subgadgets)
    
    try:
        if operation[0] == "COMBINE":
            _, g1_idx, g2_idx, rotation, splice_idx = operation
            if g1_idx >= len(network.subgadgets) or g2_idx >= len(network.subgadgets):
                return None
            
            # Validate input gadgets
            g1 = new_network.subgadgets[g1_idx]
            g2 = new_network.subgadgets[g2_idx]
            if has_duplicate_locations(g1) or has_duplicate_locations(g2):
                return None
                
            combined = network.do_combine(g1_idx, g2_idx, rotation, splice_idx)
            error_msg = validate_gadget(combined, operation)
            if error_msg:
                return None
                
            new_network.subgadgets = [g for i, g in enumerate(new_network.subgadgets) 
                                    if i not in (g1_idx, g2_idx)]
            new_network.subgadgets.append(combined)
            
        else:  # CONNECT
            _, gadget_idx, loc1, loc2 = operation
            if gadget_idx >= len(network.subgadgets):
                return None
            gadget = new_network.subgadgets[gadget_idx]
            if loc1 not in gadget.locations or loc2 not in gadget.locations:
                return None
                
            if has_duplicate_locations(gadget):
                return None
                
            network.do_connect(gadget, loc1, loc2)
            error_msg = validate_gadget(gadget, operation)
            if error_msg:
                return None
            
        return new_network
    except (ValueError, IndexError) as e:
        return None

def network_state_hash(network: GadgetNetwork) -> str:
    """Create a hash of the network's current state."""
    state_info = []
    for gadget in network.subgadgets:
        gadget_info = {
            'name': str(gadget),
            'state': gadget.getCurrentState(),
            'locations': sorted(gadget.getLocations()),
            'transitions': sorted(
                (loc1, loc2, next_state) 
                for state, transitions in gadget.transitions.items() 
                for loc1, loc2, next_state in transitions
            )
        }
        state_info.append(gadget_info)
    return str(sorted(state_info, key=lambda x: x['name']))

def search_step(network: GadgetNetwork, 
                target_gadget: Gadget,
                strategy: SearchStrategy,
                max_depth: int = 4,
                max_attempts: int = 1000,
                verbose: bool = False) -> Optional[List[Tuple]]:
    """Core search implementation supporting multiple strategies."""
    
    def check_solution(network: GadgetNetwork, path: List[Tuple]) -> bool:
        if path:
            current = network.simplify()
            if current == target_gadget:
                return True
        return False
    
    if strategy == SearchStrategy.RANDOM:
        for attempt in range(max_attempts):
            path = []
            current_network = deepcopy(network)
            visited_states = {network_state_hash(current_network)}
            
            for depth in range(max_depth):
                if check_solution(current_network, path):
                    return path
                    
                ops = get_possible_operations(current_network)
                if not ops:
                    break
                    
                operation = random.choice(ops)
                new_network = apply_operation(current_network, operation)
                
                if new_network is None:
                    continue
                    
                state_hash = network_state_hash(new_network)
                if state_hash in visited_states:
                    break
                    
                visited_states.add(state_hash)
                path.append(operation)
                current_network = new_network
                
                
        return None
        
    elif strategy == SearchStrategy.DFS:
        def dfs_helper(network: GadgetNetwork, 
                      depth: int, 
                      visited: Set[str], 
                      path: List[Tuple]) -> Optional[List[Tuple]]:
            if depth >= max_depth:
                return None
                    
            if check_solution(network, path):
                return path
                    
            state = network_state_hash(network)
            if state in visited:
                return None
                
            visited.add(state)
                
            ops = get_possible_operations(network)
            for operation in ops:
                new_network = apply_operation(network, operation)
                if new_network is None:
                    continue
                    
                result = dfs_helper(new_network, depth + 1, 
                                  visited.copy(), path + [operation])
                if result is not None:
                    return result
                        
            return None
            
        return dfs_helper(network, 0, set(), [])
        
    elif strategy == SearchStrategy.BFS:
        queue = deque([(network, [])])  # (network, path)
        visited = {network_state_hash(network)}
        
        while queue:
            current_network, path = queue.popleft()
            
            if len(path) >= max_depth:
                continue
                
            if check_solution(current_network, path):
                return path
                
            ops = get_possible_operations(current_network)
            for operation in ops:
                new_network = apply_operation(current_network, operation)
                if new_network is None:
                    continue
                    
                state_hash = network_state_hash(new_network)
                if state_hash in visited:
                    continue
                    
                visited.add(state_hash)
                queue.append((new_network, path + [operation]))
                
                
        return None

def find_simulation(initial_gadgets: List[Gadget], 
                   target_gadget: Gadget,
                   strategy: SearchStrategy = SearchStrategy.DFS,
                   max_depth: int = 10,
                   max_attempts: int = 10000,
                   verbose: bool = False) -> Optional[List[Tuple]]:
    """
    Find a sequence of operations to simulate target_gadget using initial_gadgets.
    
    Args:
        initial_gadgets: List of gadgets to start with
        target_gadget: Gadget to simulate
        strategy: SearchStrategy.DFS, BFS, or RANDOM
        max_depth: Maximum number of operations to try
        max_attempts: For random search, number of attempts to make
        verbose: If True, print detailed progress information
    """
    for state_combo in product(*[g.getStates() for g in initial_gadgets]):
            
        network = GadgetNetwork()
        for gadget, state in zip(initial_gadgets, state_combo):
            new_gadget = deepcopy(gadget)
            new_gadget.setCurrentState(state)
            network += new_gadget
            
                
        operations = search_step(network, target_gadget, strategy, 
                               max_depth, max_attempts, verbose)
            
        if operations:
            if verbose:
                print("\nSolution found!")
                print("Step-by-step construction:")
                network = GadgetNetwork()
                for gadget in initial_gadgets:
                    network += gadget
                    
                for i, op in enumerate(operations, 1):
                    print(f"\nStep {i}: {format_operation(op)}")
                    network = apply_operation(network, op)
                    print("Current configuration:")
                    for j, gadget in enumerate(network.subgadgets):
                        print(f"\nGadget {j}:")
                        print(print_gadget_definition(gadget))
                
                print("\nFinal simplified gadget:")
                final = network.simplify()
                print(print_gadget_definition(final))
                print("\nTarget gadget:")
                print(print_gadget_definition(target_gadget))
            
            return operations
            
    if verbose:
        print("\nNo solution found after trying all initial states")
    return None

def run_all_strategies(initial_gadgets, target, name="Test"):
    """Run search with all available strategies."""
    results = {}
    for strategy in SearchStrategy:
        print(f"\nTrying {strategy.value}...")
        path = find_simulation(initial_gadgets, target, strategy=strategy)
        results[strategy] = path is not None
        if path:
            print(f"Found solution with {strategy.value}:")
            for op in path:
                print(f"  {format_operation(op)}")
        else:
            print(f"No solution found with {strategy.value}")
    return results