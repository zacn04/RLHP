import sys
import os
from pathlib import Path

# Get the project root directory (RLHP)
project_root = Path(__file__).resolve().parents[3]  # Go up 3 levels
sys.path.append(str(project_root))
from enum import Enum
from itertools import product, permutations
from typing import List, Set, Dict, Tuple, Optional
from collections import deque
from copy import deepcopy
import random

# Import your gadget definitions and GadgetNetwork class
from oop.gadgets.gadgetdefs import AntiParallel2Toggle, Toggle2, Crossing2Toggle, Door, SelfClosingDoor,AntiParallelLocking2Toggle, CrossingLocking2Toggle
from oop.gadgets.gadgetlike import Gadget, GadgetNetwork

class SearchStrategy(Enum):
    DFS = "dfs"
    BFS = "bfs"
    RANDOM = "random"

def print_gadget_definition(gadget: Gadget) -> str:
    """Print a gadget's definition in a readable format."""
    output = []
    output.append(f"Gadget: {gadget.name}")
    output.append(f"Current state: {gadget.getCurrentState()}")
    output.append(f"Locations: {sorted(gadget.getLocations())}")
    output.append(f"States: {sorted(gadget.getStates())}")
    output.append("Transitions:")
    
    for state in sorted(gadget.transitions.keys()):
        output.append(f"  State {state}:")
        # Sort transitions for consistent output
        sorted_transitions = sorted(gadget.transitions[state], 
                                key=lambda x: (x[0], x[1]))
        for from_loc, to_loc, next_state in sorted_transitions:
            output.append(f"    ({from_loc} → {to_loc}) ⟹ State {next_state}")
    
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
    
    # COMBINE operations - only if we have multiple gadgets
    if len(network.subgadgets) > 1:
        for i, g1 in enumerate(network.subgadgets):
            for j, g2 in enumerate(network.subgadgets[i+1:], i+1):
                for rotation in range(len(g2.getLocations())):
                    for splice_idx in range(len(g1.getLocations())):
                        operations.append(("COMBINE", i, j, rotation, splice_idx))
    
    # CONNECT operations - for each individual gadget
    for i, gadget in enumerate(network.subgadgets):
        locs = gadget.getLocations()
        if len(locs) >= 2:  # Need at least 2 locations to connect
            n = len(locs)
            
            # Create pairs of adjacent locations using the gadget's existing order
            # which should already represent the clockwise arrangement
            for j in range(n):
                loc1 = locs[j]
                loc2 = locs[(j + 1) % n]  # Wrap around to the first location
                operations.append(("CONNECT", i, loc1, loc2))
                # Also allow the reverse direction
                operations.append(("CONNECT", i, loc2, loc1))
    
    return operations

def network_state_hash(network: GadgetNetwork) -> str:
    """Create a hash of the network's current state."""
    # For each gadget, capture: name, current state, locations, and transitions
    gadget_states = []
    
    for i, gadget in enumerate(network.subgadgets):
        # Get transitions as tuples for hashing
        transitions = []
        for state, trans_list in gadget.transitions.items():
            # Sort for consistent hashing
            for loc1, loc2, next_state in sorted(trans_list):
                transitions.append((state, loc1, loc2, next_state))
        
        gadget_info = {
            'name': gadget.name,
            'state': gadget.getCurrentState(),
            'locations': tuple(sorted(gadget.getLocations())),
            'transitions': tuple(sorted(transitions))
        }
        gadget_states.append(str(gadget_info))
    
    # Sort for consistent hashing regardless of gadget order
    return str(sorted(gadget_states))
def apply_operation(network: GadgetNetwork, operation: Tuple) -> Optional[GadgetNetwork]:
    """Apply an operation to the network with validation and simplification."""
    # Make a deep copy to avoid modifying the original
    new_network = GadgetNetwork()
    new_network.subgadgets = deepcopy(network.subgadgets)
    
    try:
        if operation[0] == "COMBINE":
            _, g1_idx, g2_idx, rotation, splice_idx = operation
            
            # Validate indices
            if g1_idx >= len(new_network.subgadgets) or g2_idx >= len(new_network.subgadgets):
                return None
            
            # Perform the combination
            combined = new_network.do_combine(g1_idx, g2_idx, rotation, splice_idx)
            
            # Update the network's gadgets list
            # Remove the original gadgets (in reverse order to avoid index shifting)
            if g2_idx > g1_idx:
                new_network.subgadgets.pop(g2_idx)
                new_network.subgadgets.pop(g1_idx)
            else:
                new_network.subgadgets.pop(g1_idx)
                new_network.subgadgets.pop(g2_idx)
            
            # Add the combined gadget
            new_network.subgadgets.append(combined)
            
        elif operation[0] == "CONNECT":
            _, gadget_idx, loc1, loc2 = operation
            
            # Validate index
            if gadget_idx >= len(new_network.subgadgets):
                return None
                
            gadget = new_network.subgadgets[gadget_idx]
            
            # Validate locations
            if loc1 not in gadget.getLocations() or loc2 not in gadget.getLocations():
                return None
                
            # Perform the connection
            new_network.do_connect(gadget, loc1, loc2)
        
        # Simplify any gadgets with empty states if we have a single gadget
        if len(new_network.subgadgets) == 1:
            # The do_combine method should now handle simplification internally
            # But for gadgets that went through connect operations, we might need to simplify
            simplified = new_network.simplify()
            if simplified:
                new_network.subgadgets = [simplified]
            
        return new_network
        
    except Exception as e:
        print(f"Error applying operation {operation}: {e}")
        return None

def check_solution(gadget: Gadget, target_gadget: Gadget) -> bool:
    """Check if a gadget is equivalent to the target gadget."""
    try:
        # The __eq__ method should now correctly handle the comparison
        # even with different state structures due to combinations
        return gadget == target_gadget
    except Exception as e:
        print(f"Error comparing gadgets: {e}")
        return False

def find_simulation(initial_gadgets: List[Gadget], 
                   target_gadget: Gadget,
                   strategy: SearchStrategy = SearchStrategy.BFS,
                   max_depth: int = 5,
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
    if verbose:
        print("Target gadget:")
        print(print_gadget_definition(target_gadget))
        print(f"Searching for solution using {strategy.value} with max depth {max_depth}")
    
    # Try all combinations of initial states
    for state_combo in product(*[g.getStates() for g in initial_gadgets]):
        if verbose:
            print(f"Trying initial states: {state_combo}")
            
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
                demonstrate_solution(initial_gadgets, state_combo, operations, target_gadget, verbose)
            return operations
            
    if verbose:
        print("\nNo solution found after trying all initial states")
    return None

def demonstrate_solution(initial_gadgets: List[Gadget], 
                        state_combo: Tuple, 
                        operations: List[Tuple], 
                        target_gadget: Gadget,
                        verbose: bool = True) -> None:
    """Demonstrate the found solution step by step."""
    if verbose:
        print("\nStep-by-step solution:")
        
        # Set up the initial network
        network = GadgetNetwork()
        for i, (gadget, state) in enumerate(zip(initial_gadgets, state_combo)):
            new_gadget = deepcopy(gadget)
            new_gadget.setCurrentState(state)
            network += new_gadget
            print(f"\nInitial Gadget {i} (state {state}):")
            print(print_gadget_definition(new_gadget))
        
        # Apply each operation
        for i, op in enumerate(operations, 1):
            print(f"\nStep {i}: {format_operation(op)}")
            
            if op[0] == "COMBINE":
                _, g1_idx, g2_idx, rotation, splice_idx = op
                
                # Show pre-combine state
                print(f"  Combining gadget {g1_idx} with gadget {g2_idx}")
                print(f"  Gadget {g1_idx} locations: {network.subgadgets[g1_idx].getLocations()}")
                print(f"  Gadget {g2_idx} locations: {network.subgadgets[g2_idx].getLocations()}")
                
                # Perform the combination
                combined = network.do_combine(g1_idx, g2_idx, rotation, splice_idx)
                
                # Update the network's gadgets list
                # Remove the original gadgets (in reverse order to avoid index shifting)
                if g2_idx > g1_idx:
                    network.subgadgets.pop(g2_idx)
                    network.subgadgets.pop(g1_idx)
                else:
                    network.subgadgets.pop(g1_idx)
                    network.subgadgets.pop(g2_idx)
                
                # Add the combined gadget
                network.subgadgets.append(combined)
                
                print(f"  Combined gadget locations: {combined.getLocations()}")
                
            else:  # CONNECT
                _, gadget_idx, loc1, loc2 = op
                gadget = network.subgadgets[gadget_idx]
                
                # Show pre-connect state
                print(f"  Connecting locations {loc1} and {loc2} in gadget {gadget_idx}")
                print(f"  Gadget locations before: {gadget.getLocations()}")
                
                # Perform the connection
                network.do_connect(gadget, loc1, loc2)
                
                print(f"  Gadget locations after: {gadget.getLocations()}")
            
            # Show the updated state of each gadget
            print("Current network configuration:")
            for j, gadget in enumerate(network.subgadgets):
                print(f"Gadget {j}:")
                print(print_gadget_definition(gadget))
        
        # Simplify and check the final result
        if len(network.subgadgets) == 1:
            simplified = network.simplify()
            if simplified:
                print("\nFinal simplified gadget:")
                print(print_reachable_states(simplified))
                print("\nTarget gadget:")
                print(print_reachable_states(target_gadget))
                print(f"Equal: {simplified == target_gadget}")
            else:
                print("\nFinal gadget (not simplified):")
                print(print_reachable_states(network.subgadgets[0]))
                print("\nTarget gadget:")
                print(print_reachable_states(target_gadget))
                print(f"Equal: {network.subgadgets[0] == target_gadget}")
        else:
            print(f"\nWarning: Final network has {len(network.subgadgets)} gadgets (expected 1)")

def search_step(initial_network: GadgetNetwork, 
                target_gadget: Gadget,
                strategy: SearchStrategy,
                max_depth: int = 5,
                max_attempts: int = 1000,
                verbose: bool = False) -> Optional[List[Tuple]]:
    """Core search implementation supporting multiple strategies."""
    
    if strategy == SearchStrategy.RANDOM:
        # Random search implementation
        for attempt in range(max_attempts):
            current_network = deepcopy(initial_network)
            operations_applied = []
            visited_states = {network_state_hash(current_network)}
            
            for depth in range(max_depth):
                ops = get_possible_operations(current_network)
                if not ops:
                    break
                    
                operation = random.choice(ops)
                if verbose and (attempt % 100 == 0 or attempt < 10):
                    print(f"Attempt {attempt}, step {depth}: Trying {format_operation(operation)}")
                
                new_network = apply_operation(current_network, operation)
                
                if new_network is None:
                    continue
                    
                state_hash = network_state_hash(new_network)
                if state_hash in visited_states:
                    continue
                    
                visited_states.add(state_hash)
                operations_applied.append(operation)
                current_network = new_network
                
                # Check if we've found a solution
                if len(current_network.subgadgets) == 1:
                    # Get the simplified gadget if needed
                    gadget = current_network.subgadgets[0]
                    if check_solution(gadget, target_gadget):
                        if verbose:
                            print(f"Solution found on attempt {attempt}!")
                        return operations_applied
            
        return None
        
    elif strategy == SearchStrategy.DFS:
        # DFS implementation
        stack = [(initial_network, [])]  # (network, operations_applied)
        visited = {network_state_hash(initial_network)}
        nodes_explored = 0
        
        while stack:
            current_network, operations_applied = stack.pop()
            nodes_explored += 1
            
            if len(operations_applied) >= max_depth:
                continue
                
            if verbose and nodes_explored % 1000 == 0:
                print(f"DFS: Explored {nodes_explored} nodes, current depth {len(operations_applied)}")
                
            ops = get_possible_operations(current_network)
            for operation in reversed(ops):  # Reversed for proper DFS order with stack
                new_network = apply_operation(current_network, operation)
                
                if new_network is None:
                    continue
                    
                state_hash = network_state_hash(new_network)
                if state_hash in visited:
                    continue
                    
                visited.add(state_hash)
                new_operations = operations_applied + [operation]
                
                # Check if we've found a solution
                if len(new_network.subgadgets) == 1:
                    # Get the simplified gadget if needed
                    gadget = new_network.subgadgets[0]
                    if check_solution(gadget, target_gadget):
                        if verbose:
                            print(f"Solution found after exploring {nodes_explored} nodes!")
                        return new_operations
                
                stack.append((new_network, new_operations))
                
        return None
        
    elif strategy == SearchStrategy.BFS:
        # BFS implementation
        queue = deque([(initial_network, [])])  # (network, operations_applied)
        visited = {network_state_hash(initial_network)}
        nodes_explored = 0
        
        while queue:
            current_network, operations_applied = queue.popleft()
            nodes_explored += 1
            
            if len(operations_applied) >= max_depth:
                continue
                
            if verbose and nodes_explored % 1000 == 0:
                print(f"BFS: Explored {nodes_explored} nodes, current depth {len(operations_applied)}")
                
            ops = get_possible_operations(current_network)
            for operation in ops:
                new_network = apply_operation(current_network, operation)
                
                if new_network is None:
                    continue
                    
                state_hash = network_state_hash(new_network)
                if state_hash in visited:
                    continue
                    
                visited.add(state_hash)
                new_operations = operations_applied + [operation]
                
                # Check if we've found a solution
                if len(new_network.subgadgets) == 1:
                    # The apply_operation function should have already simplified the gadget
                    gadget = new_network.subgadgets[0]
                    if check_solution(gadget, target_gadget):
                        if verbose:
                            print(f"Solution found after exploring {nodes_explored} nodes!")
                        return new_operations
                
                queue.append((new_network, new_operations))
                
        return None


def find_simulation(initial_gadgets: List[Gadget], 
                   target_gadget: Gadget,
                   strategy: SearchStrategy = SearchStrategy.BFS,
                   max_depth: int = 5,
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
    if verbose:
        print("Target gadget:")
        print(print_gadget_definition(target_gadget))
        print(f"Searching for solution using {strategy.value} with max depth {max_depth}")
    
    # Try all combinations of initial states
    for state_combo in product(*[g.getStates() for g in initial_gadgets]):
        if verbose:
            print(f"Trying initial states: {state_combo}")
            
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
                demonstrate_solution(initial_gadgets, state_combo, operations, target_gadget, verbose)
            return operations
            
    if verbose:
        print("\nNo solution found after trying all initial states")
    return None






def print_reachable_states(gadget: Gadget, verbose: bool = True) -> str:
    """
    Analyze and print only the states reachable from the current state.
    
    Args:
        gadget: The gadget to analyze
        verbose: If True, print details during analysis
    
    Returns:
        A string with the formatted reachable states and transitions
    """
    # Start BFS from the current state
    current_state = gadget.getCurrentState()
    if verbose:
        print(f"Starting analysis from state: {current_state}")
    
    # Track visited states and build reachable transitions
    visited = set([current_state])
    reachable_transitions = {}
    queue = deque([current_state])
    
    # Perform BFS
    while queue:
        state = queue.popleft()
        
        # Skip if this state has no transitions
        if state not in gadget.transitions:
            continue
            
        # Initialize state transitions if not already done
        if state not in reachable_transitions:
            reachable_transitions[state] = []
            
        # Process all transitions from this state
        for from_loc, to_loc, next_state in gadget.transitions[state]:
            # Add this transition to our reachable set
            reachable_transitions[state].append((from_loc, to_loc, next_state))
            
            # If next state not visited, add to queue
            if next_state not in visited:
                visited.add(next_state)
                queue.append(next_state)
    
    # Build the output string
    output = []
    output.append(f"Gadget: {gadget.name}")
    output.append(f"Current state: {current_state}")
    output.append(f"Reachable locations: {sorted(set(loc for state in reachable_transitions for from_loc, to_loc, _ in reachable_transitions[state] for loc in (from_loc, to_loc)))}")
    output.append(f"Reachable states: {sorted(visited)}")
    output.append("Reachable transitions:")
    
    # Print transitions for each reachable state
    for state in sorted(reachable_transitions.keys()):
        output.append(f"  State {state}:")
        sorted_transitions = sorted(reachable_transitions[state], 
                                  key=lambda x: (x[0], x[1]))
        for from_loc, to_loc, next_state in sorted_transitions:
            output.append(f"    ({from_loc} → {to_loc}) ⟹ State {next_state}")
    
    result = "\n".join(output)
    
    return result

"""Run a search to find how two AntiParallel2Toggle gadgets can simulate a CrossingLocking2Toggle."""
    # Create instances of the gadgets
ap2t1 = AntiParallel2Toggle()  # First instance
ap2t2 = AntiParallel2Toggle()  # Second instance
target = Crossing2Toggle()
    
print("Searching for solution to create CrossingLocking2Toggle from two AntiParallel2Toggles...")
operations = find_simulation([ap2t1, ap2t2], target, 
                              strategy=SearchStrategy.BFS,
                              max_depth=6, 
                              verbose=True)
    


if operations:
    print("\nSolution found!")
    print("Operations:", [format_operation(op) for op in operations])
else:
    print("No solution found.")



