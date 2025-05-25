import ctypes
import numpy as np
import os
from oop.gadgets.gadgetdefs import *
from oop.gadgets.gadgetlike import GadgetNetwork

# Load the C++ library
lib_path = os.path.join(os.path.dirname(__file__), '..', 'jeffrey', 'motion-planning-gadget-search', 'build', 'libgadget_equiv.so')
lib = ctypes.CDLL(lib_path)

# Define the C function signature
lib.are_gadgets_equivalent.argtypes = [
    ctypes.POINTER(ctypes.c_int),  # locations1
    ctypes.c_int,                  # num_locations1
    ctypes.POINTER(ctypes.c_int),  # states1
    ctypes.c_int,                  # num_states1
    ctypes.POINTER(ctypes.c_int),  # transitions1
    ctypes.c_int,                  # num_transitions1
    ctypes.c_int,                  # current_state1
    ctypes.POINTER(ctypes.c_int),  # locations2
    ctypes.c_int,                  # num_locations2
    ctypes.POINTER(ctypes.c_int),  # states2
    ctypes.c_int,                  # num_states2
    ctypes.POINTER(ctypes.c_int),  # transitions2
    ctypes.c_int,                  # num_transitions2
    ctypes.c_int,                  # current_state2
    ctypes.c_int                   # alphabet_size
]
lib.are_gadgets_equivalent.restype = ctypes.c_bool

def are_gadgets_equivalent(gadget1, gadget2, alphabet_size=4):
    """
    Check if two gadgets are equivalent using Jeffrey's automaton-based equivalence checking.
    
    Args:
        gadget1: First gadget (must have locations, states, transitions, and current_state attributes)
        gadget2: Second gadget (must have locations, states, transitions, and current_state attributes)
        alphabet_size: Size of the alphabet (default 4)
        
    Returns:
        bool: True if the gadgets are equivalent, False otherwise
    """
    # Convert gadget1 to C arrays
    locations1 = (ctypes.c_int * len(gadget1.locations))(*gadget1.locations)
    states1 = (ctypes.c_int * len(gadget1.states))(*gadget1.states)
    
    # Flatten transitions for gadget1
    transitions1 = []
    for state, trans_list in gadget1.transitions.items():
        for trans in trans_list:
            transitions1.extend([state, trans[0], trans[1], trans[2]])
    transitions1 = (ctypes.c_int * len(transitions1))(*transitions1)
    
    # Convert gadget2 to C arrays
    locations2 = (ctypes.c_int * len(gadget2.locations))(*gadget2.locations)
    states2 = (ctypes.c_int * len(gadget2.states))(*gadget2.states)
    
    # Flatten transitions for gadget2
    transitions2 = []
    for state, trans_list in gadget2.transitions.items():
        for trans in trans_list:
            transitions2.extend([state, trans[0], trans[1], trans[2]])
    transitions2 = (ctypes.c_int * len(transitions2))(*transitions2)
    
    # Call the C function
    return lib.are_gadgets_equivalent(
        locations1, len(gadget1.locations),
        states1, len(gadget1.states),
        transitions1, len(transitions1),
        gadget1.current_state,
        locations2, len(gadget2.locations),
        states2, len(gadget2.states),
        transitions2, len(transitions2),
        gadget2.current_state,
        alphabet_size
    )

def test_equivalence():
    """Test the equivalence checking with known equivalent gadgets."""
    # Test 1: Two identical ParallelLocking2Toggle gadgets
    pl2t1 = ParallelLocking2Toggle()
    pl2t2 = ParallelLocking2Toggle()
    assert are_gadgets_equivalent(pl2t1, pl2t2), "Identical PL2T gadgets should be equivalent"
    
    # Test 2: AP2T combined with itself should be equivalent to C2T
    net = GadgetNetwork()
    ap2t1 = AntiParallel2Toggle()
    net += ap2t1
    ap2t2 = AntiParallel2Toggle()
    ap2t2.setCurrentState(1)
    net += ap2t2
    combined = net.do_combine(0, 1, rotation=0, splicing_index=3)
    net += combined
    net.connect(2, 0, 4)
    net.connect(2, 2, 6)
    res = net.simplify()
    
    c2t = Crossing2Toggle()
    assert are_gadgets_equivalent(res, c2t), "AP2T combined with itself should be equivalent to C2T"
    
    # Test 3: CL2T combined with itself should be equivalent to PL2T
    net = GadgetNetwork()
    cl2t1 = CrossingLocking2Toggle()
    net += cl2t1
    cl2t2 = CrossingLocking2Toggle()
    net += cl2t2
    combined = net.do_combine(0, 1, rotation=0, splicing_index=1)
    net += combined
    net.connect(2, 6, 5)
    net.connect(2, 1, 2)
    res = net.simplify()
    
    pl2t = ParallelLocking2Toggle()
    assert are_gadgets_equivalent(res, pl2t), "CL2T combined with itself should be equivalent to PL2T"
    
    print("All equivalence tests passed!")

if __name__ == "__main__":
    test_equivalence() 