from itertools import product, permutations
from typing import List, Set, Dict, Tuple
from copy import deepcopy
from oop.gadgets.gadgetdefs import AntiParallel2Toggle, Crossing2Toggle
from oop.gadgets.gadgetlike import GadgetNetwork, Gadget

def get_possible_operations(network: GadgetNetwork) -> List[Tuple]:
    operations = []
    print(network.subgadgets, len(network.subgadgets))
    
    if len(network.subgadgets) > 1:
        for i, g1 in enumerate(network.subgadgets):
            for j, g2 in enumerate(network.subgadgets[i+1:], i+1):
                for rotation in range(len(g2.getLocations())):
                    for splice_idx in range(len(g1.getLocations())):
                        operations.append(("COMBINE", i, j, rotation, splice_idx))
    
    for i, gadget in enumerate(network.subgadgets):
        locs = gadget.getLocations()
        print(gadget, locs)
        for loc1, loc2 in permutations(locs, 2):
            operations.append(("CONNECT", i, loc1, loc2))
    
    return operations

def network_state_hash(network: GadgetNetwork) -> str:
    """Create a hash that captures gadget states and their connections"""
    state_info = []
    for gadget in network.subgadgets:
        gadget_info = {
            'name': str(gadget),
            'state': gadget.getCurrentState(),
            'locations': sorted(gadget.getLocations()),
            'transitions': sorted(
                (loc1, loc2) 
                for state in gadget.transitions.values() 
                for (loc1, loc2) in state.keys()
            )
        }
        state_info.append(gadget_info)
    return str(state_info)

def apply_operation(network: GadgetNetwork, operation: Tuple):
    if operation[0] == "COMBINE":
        _, g1_idx, g2_idx, rotation, splice_idx = operation
        if g1_idx >= len(network.subgadgets) or g2_idx >= len(network.subgadgets):
            return None
            
        print(f"Before combine - g1 state: {network.subgadgets[g1_idx].getCurrentState()}, g2 state: {network.subgadgets[g2_idx].getCurrentState()}")
        combined_name = f"Combined({network.subgadgets[g1_idx].name}+{network.subgadgets[g2_idx].name})"
        if any(g.name == combined_name for g in network.subgadgets):
            return None
    else:
        _, gadget_idx, loc1, loc2 = operation
        if gadget_idx >= len(network.subgadgets):
            return None
        gadget = network.subgadgets[gadget_idx]
        if loc1 not in gadget.locations or loc2 not in gadget.locations:
            return None

    new_network = GadgetNetwork()
    new_network.subgadgets = deepcopy(network.subgadgets)
    new_network.operations = []
    
    try:
        if operation[0] == "COMBINE":
            combined = network.do_combine(g1_idx, g2_idx, rotation, splice_idx)
            print(f"After combine - new state: {combined.getCurrentState()}")
            new_network.subgadgets = [g for i, g in enumerate(new_network.subgadgets) 
                                    if i not in (g1_idx, g2_idx)]
            new_network.subgadgets.append(combined)
        else:
            network.do_connect(new_network.subgadgets[gadget_idx], loc1, loc2)
            
        return new_network
    except (ValueError, IndexError):
        return None

def dfs_find_simulation(initial_network: GadgetNetwork, 
                       target_gadget: Gadget, 
                       max_depth: int = 4) -> List[Tuple]:
    def dfs_helper(network: GadgetNetwork, 
                  depth: int, 
                  visited: Set[str], 
                  path: List[Tuple]) -> List[Tuple]:
        print(f"\n>>> Entering depth {depth}")
        print(f">>> Current path: {path}")
        
        if depth >= max_depth:
            print(f">>> Hit max depth {depth}")
            return None
                
        state = network_state_hash(network)
        print(f">>> Got state hash: {state}")
        print(f">>> Visited states: {visited}")
        if state in visited:
            print(f">>> State already visited at depth {depth}")
            return None
        visited.add(state)

        print(f"\nDepth {depth}, Current path: {path}")
        print(f"Current gadgets: {[str(g) for g in network.subgadgets]}")

        if path:
            current = network.simplify()
            print(f"After simplify: {current}")
            if current == target_gadget:
                return path

        ops = get_possible_operations(network)
        print(f"Available operations at depth {depth}: {ops}")

        for operation in ops:
            print(f"\nTrying at depth {depth}: {operation}")
            new_network = apply_operation(network, operation)
            if new_network is None:
                print(f"Operation {operation} failed to create valid network")
                continue
                
            new_visited = visited.copy()
            
            print(f"Recursing to depth {depth+1} with operation {operation}")
            result = dfs_helper(new_network, depth + 1, new_visited, path + [operation])
            if result is not None:
                return result
            print(f"No solution found in branch at depth {depth} after {operation}")
                    
        print(f"Backtracking from depth {depth}")
        return None

    visited = set()
    return dfs_helper(initial_network, 0, visited, [])

def find_simulation(initial_gadgets: List[Gadget], target_gadget: Gadget) -> List[Tuple]:
    gadget_states = []
    for gadget in initial_gadgets:
        gadget_states.append(gadget.getStates())
    
    for state_combo in product(*gadget_states):
        print(f"\nTrying states: {state_combo}")
        network = GadgetNetwork()
        
        for i, (gadget, state) in enumerate(zip(initial_gadgets, state_combo)):
            print(f"Setting gadget {i} to state {state}")
            new_gadget = deepcopy(gadget)
            new_gadget.setCurrentState(state)
            print(f"Gadget {i} state after setting: {new_gadget.getCurrentState()}")
            network += new_gadget
            
        print(f"Initial gadget states: {[g.getCurrentState() for g in network.subgadgets]}")
        operations = dfs_find_simulation(network, target_gadget)
        if operations:
            return operations
            
    return None

if __name__ == "__main__":

    ap2t1 = AntiParallel2Toggle()
    ap2t2 = AntiParallel2Toggle()
    initial_gadgets = [ap2t1, ap2t2]

    target = Crossing2Toggle()
    
    operations = find_simulation(initial_gadgets, target)
    
    print("\nFinal Results:")
    if operations:
        print("Found solution:")
        network = GadgetNetwork()
        for gadget in initial_gadgets:
            network += gadget
        print(f"\nInitial states: {[g.getCurrentState() for g in network.subgadgets]}")
        
        for i, op in enumerate(operations):
            print(f"\nStep {i+1}: {op}")
            network = apply_operation(network, op)
            print(f"Gadget states after operation: {[g.getCurrentState() for g in network.subgadgets]}")
            print(f"Current gadgets: {[str(g) for g in network.subgadgets]}")
    else:
        print("No solution found")