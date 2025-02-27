'''
Author: Zac Nwogwugwu, 2025
This defines our toolkit for gadgets with the new transition (triples) format.
'''
from oop.gadgets.gadgetlike import Gadget, GadgetNetwork

# LOCATIONS MUST BE CLOCKWISE!
class ParallelLocking2Toggle(Gadget):
    """
    Parallel Locking 2 Toggle Gadget. (PL2T)
    """
    def __init__(self):
        locations = [0, 1, 2, 3]
        states = [0, 1, 2]
        name = "Locking 2 Toggle"
        transitions = {
            0: [(0, 1, 1), (3, 2, 2)],
            1: [(1, 0, 0)],
            2: [(2, 3, 0)]
        }
        current_state = 0
        super().__init__(name, locations, states, transitions, current_state)

class Door(Gadget):
    """
    Door Gadget.
    """
    def __init__(self):
        locations = [0, 1, 2, 3]
        states = [0, 1]
        name = "Door"
        transitions = {
            0: [(0, 1, 1)],
            1: [(2, 3, 0), (1, 0, 1), (0, 1, 1)]  # traversing an open door stays open, can still close it.
        }
        current_state = 0
        super().__init__(name, locations, states, transitions, current_state)

class SelfClosingDoor(Gadget):
    """
    Self Closing Door Gadget.
    """
    def __init__(self):
        locations = [0, 1, 2, 3]
        states = [0, 1]  # State 0 is closed. State 1 is open.
        name = "Self Closing Door"
        transitions = {
            0: [(0, 1, 1)],
            1: [(2, 3, 0), (1, 0, 0)]  # traversing an open door closes it
        }
        current_state = 0
        super().__init__(name, locations, states, transitions, current_state)

class Toggle2(Gadget):
    """
    Parallel 2 Toggle Gadget.
    """
    # both tunnels flip when you go through.
    def __init__(self):
        locations = [0, 1, 2, 3]
        states = [0, 1]
        name = "2 Toggle"
        transitions = {
            0: [(0, 1, 1), (3, 2, 1)],
            1: [(1, 0, 0), (2, 3, 0)],
        }
        current_state = 0
        super().__init__(name, locations, states, transitions, current_state)

class AntiParallel2Toggle(Gadget):
    """
    Anti-Parallel 2 Toggle Gadget.
    """
    def __init__(self):
        locations = [0, 1, 2, 3]
        states = [0, 1]
        name = "AntiParallel 2 Toggle"
        transitions = {
            0: [(0, 1, 1), (2, 3, 1)],
            1: [(1, 0, 0), (3, 2, 0)]
        }
        current_state = 0
        super().__init__(name, locations, states, transitions, current_state)

class Crossing2Toggle(Gadget):
    """
    Crossing 2 Toggle Gadget.
    """
    def __init__(self):
        name = "Crossing2Toggle"
        locations = [0, 1, 2, 3]
        states = [0, 1]
        transitions = {
            0: [(0, 2, 1), (3, 1, 1)],
            1: [(2, 0, 0), (1, 3, 0)]
        }
        current_state = 0
        super().__init__(name, locations, states, transitions, current_state)

class AntiParallelLocking2Toggle(Gadget):
    """
    Anti-Parallel Locking 2 Toggle Gadget.
    """
    def __init__(self):
        locations = [0, 1, 2, 3]
        states = [0, 1, 2]
        name = "APL2T"
        transitions = {
            0: [(0, 1, 1), (2, 3, 2)],
            1: [(1, 0, 0)],
            2: [(3, 2, 0)]
        }
        current_state = 0
        super().__init__(name, locations, states, transitions, current_state)

class CrossingLocking2Toggle(Gadget):
    """
    Crossing Locking 2 Toggle Gadget.
    """
    def __init__(self):
        locations = [0, 1, 2, 3]
        states = [0, 1, 2]
        name = "CL2T"
        transitions = {
            0: [(0, 2, 1), (3, 1, 2)],
            1: [(2, 0, 0)],
            2: [(1, 3, 0)]
        }
        current_state = 0
        super().__init__(name, locations, states, transitions, current_state)


