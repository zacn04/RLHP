"""
Author: Zac Nwogwugwu, 2025
"""
from gadget import Gadget
def do_simulate(gadgets: list[Gadget], combinations, target: Gadget) -> bool:
    """
    Does a combination of gadgets (through connecting or combining) simulate the target?
    Returns TRUE if so.
    """
    return True



# Need some way of defining these combinations...

# This forms the basis of the reinforcement learning step.

# can a combination just be initialising another gadget with a superimposition?

def connect(gadget1: Gadget, gadget2: Gadget, location1: int, location2: int) -> Gadget:
    """
    This operation connects location1 of gadget1 to location2 of gadget2 and returns a new gadget.
    """

    #For this we need to know which traversals location1 and 2 are involved in...
    #Naively we can iterate through gadget1's states and see which have location 1 as a "out location"
    #and the same for gadget2, seeing which have location 2 as an "in location", then keeping track of those
    #create a new gadget.

    #Q for Jayson: When connecting 2 gadget's locations, how does this impact state changes?

    new_transitions = []

    out_transitions = [
        (start, end)
        for out_state in gadget1.states
        for (start, end) in gadget1.states[out_state]
        if end == location1
    ]
    in_transitions = [
        (start, end)
        for in_state in gadget2.states
        for (start, end) in gadget2.states[in_state]
        if start == location1
    ]

    for (start1, end1) in out_transitions:
        for (start2, end2) in in_transitions:
            new_transitions.append((start1, end2))

    # TODO: push into the GadgetNetwork class.
    



