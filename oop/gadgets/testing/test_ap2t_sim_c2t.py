import sys
import os
from pathlib import Path

# Get the project root directory (RLHP)
project_root = Path(__file__).resolve().parents[3]  # Go up 3 levels
sys.path.append(str(project_root))

from oop.gadgets.gadgetdefs import AntiParallel2Toggle, Crossing2Toggle
from oop.gadgets.gadgetlike import GadgetNetwork, Gadget
from copy import deepcopy

def test_ap2t_sim_c2t():
    # Create a fresh network
    net = GadgetNetwork()
    
    # Add the first AP2T in state 0
    ap2t1 = AntiParallel2Toggle()
    net += ap2t1
    
    # Add the second AP2T in state 1
    ap2t2 = AntiParallel2Toggle()
    ap2t2.setCurrentState(1)
    net += ap2t2
    
    print("Initial network:")
    print(net)
    
    # Step 1: Combine the gadgets directly
    combined = net.do_combine(0, 1, rotation=0, splicing_index=1, reflect=False)
    
    print(f"Combined gadget created: {combined.name}")
    
    # Replace the original gadgets with the combined one
    # This is critical - we need to clear the original gadgets
    net.subgadgets = [combined]
    
    net.do_connect(combined, 1, 5)
    
    net.do_connect(combined, 2, 6)
    
    # Simplify the network
    simplified = net.simplify()
    
    # Print the final state of the simplified gadget
    print("\nFinal gadget locations:", simplified.getLocations())
    
    # Create a target Crossing2Toggle
    target = Crossing2Toggle()
    
    # Compare the simplified gadget with the target
    print("\nComparison with target:")
    print(f"Simplified transitions: {simplified.transitions}")
    print(f"Target transitions: {target.transitions}")
    
    # Check equality
    equal = simplified == target
    print(f"Gadgets are equal: {equal}")
    
    if not equal:
        # Print detailed gadget definitions for comparison
        print("\nDetailed Simplified Gadget:")
        print_gadget(simplified)
        
        print("\nDetailed Target Gadget:")
        print_gadget(target)
    
    return equal

def print_gadget(gadget: Gadget):
    """Print a gadget's definition in a readable format."""
    print(f"Name: {gadget.name}")
    print(f"Current State: {gadget.getCurrentState()}")
    print(f"Locations: {gadget.getLocations()}")
    print(f"States: {gadget.getStates()}")
    print("Transitions:")
    
    for state in sorted(gadget.transitions.keys()):
        print(f"  State {state}:")
        for from_loc, to_loc, next_state in sorted(gadget.transitions[state]):
            print(f"    ({from_loc} → {to_loc}) ⟹ State {next_state}")

if __name__ == "__main__":
    result = test_ap2t_sim_c2t()
    print(f"\nTest passed: {result}")