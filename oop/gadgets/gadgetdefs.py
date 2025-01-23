'''
Author: Zac Nwogwugwu, 2024

This will most likely be defining our toolkit for gadgets.
'''

from oop.gadgets.gadgetlike import Gadget
#LOCATIONS MUST BE CLOCKWISE!

class Toggle2Locking(Gadget):
    """
    Locking 2 Toggle Gadget.
    """
    def __init__(self):
        locations = [0,1,2,3]
        states = [0,1,2]
        name = "Locking 2 Toggle"
        transitions = {
            0: {(0, 1): 1, (3, 2): 2},
            1: {(1, 0): 0},
            2: {(2, 3): 0}
        }
        current_state = 0
        super().__init__(name, locations, states, transitions, current_state)

class Door(Gadget):
    """
    Door Gadget.
    """
    def __init__(self):
        locations = [0,1,2,3]
        states = [0,1]
        name = "Door"
        transitions = {
            0: {(0,1): 1},
            1: {(2,3): 0, (1,0): 1} #traversing an open door stays open, can still close it.
        }
        current_state = 0
        super().__init__(name, locations, states, transitions, current_state)

class SelfClosingDoor(Gadget):
    #basically same as door but any traversal of the open door closes it.
    """
    Self Closing Door Gadget.
    """
    def __init__(self):
        locations = [0,1,2,3]
        states = [0,1] #State 0 is closed. State 1 is open.
        name = "Self Closing Door"
        transitions = {
            0: {(0,1): 1},
            1: {(2,3): 0, (1,0): 0} #traversing an open door closes it, in any state
        }
        current_state = 0
        super().__init__(name, locations, states, transitions, current_state)

class Toggle2(Gadget):
    """
    2 Toggle Gadget.
    """
    #both tunnels flip when you go through.
    def __init__(self):
        locations = [0,1,2,3]
        states = [0,1]
        name = "2 Toggle"
        transitions = {
            0: {(0, 1): 1, (3, 2): 1},
            1: {(1, 0): 0, (2, 3): 0},
        }
        current_state = 0
        super().__init__(name, locations, states, transitions, current_state)

class AntiParallel2Toggle(Gadget):
    """
    Anti-Parallel 2 Toggle Gadget.
    """
    def __init__(self):
        locations = [0,1,2,3]
        states = [0,1,2]
        name = "AntiParallel 2 Toggle"
        transitions = {
            0: {(0, 1): 2, (2, 3): 1},
            1: {(1, 0): 0},
            2: {(3, 2): 0}
        }
        current_state = 0
        super().__init__(name, locations, states, transitions, current_state)
