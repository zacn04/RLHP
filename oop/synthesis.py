"""
Author: Zac Nwogwugwu, 2025
"""
from gadgets import Gadget
import random

def generate_gadget(name, num_states, num_transitions, num_locations):
    """
    Builds a random gadget named name with num_states states and num_transitions transitions per state.
    """
    gadget = Gadget()

    gadget.name = name
    gadget.states = list(range(num_states))
    gadget.locations = list(range(num_locations))

    transitions = {i:{} for i in gadget.getStates()}

    for _ in range(num_transitions):
        # pick a random tuple of transitions from a random state to a random other one
        # need some way of ensuring that our transitions make sense. 
        pass
        
    return gadget

#lower priority... but also how we are generating synthetic data!


