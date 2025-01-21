"""
Author: Zac Nwogwugwu 2025

Handling the logic for gadget networks.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

from gadget import Gadget

class GadgetLike:
    """
    The base class for gadgets/gadget networks.
    """
    @abstractmethod
    def get_locations(self) -> List[int]:
        pass

    @abstractmethod
    def get_states(self) -> List[int]:
        pass

    @abstractmethod
    def get_transitions(self) -> Dict[int, Dict[Tuple[int, int], int]]:
        """
        Returns transitions in a canonical format.
        e.g.
        {
        0 : {(1,2): 1} 
        }

        The above is a transition in state 0 from location 1 to location 2, 
        which changes the gadget state to 1
        """
        pass

class GadgetNetwork(GadgetLike):
    def __init__(self, name="GadgetNetwork"):
        """
        The environment in which gadgets will be manipulated.
        """
        self.name = name
        self.subgadgets = []
        self.connections = []

    def connect(self, gadget1, gadget2, loc1, loc2):
        self.connections.append(("CONNECT", gadget1, gadget2, loc1, loc2))

    def combine(self, gadget1, gadget2, rotation, splicing_index):
        # Rotate right side gadget per rotation
        # then create a new gadget with the same transitions 
        # but with a rearrangement of locations per splicing index

        # Rotation
        # Seems trivial, just maintain states but change each location modulo rotation?

        #Actually i dont think location really impacts state changes...

        modulo_index = len(gadget2.locations)
        rotated_locations = [(location + rotation) % modulo_index for location in gadget2.locations]

        #renames the locations 
        rotated_gadget2 = Gadget(
            name=f"{rotation}-Rotated {gadget2.name}",
            locations=rotated_locations,
            states=gadget2.states, #assuming the rorations have no bearings of states
            transitions=gadget2.transitions,
            current_state=gadget2.current_state
            )
        
        new_locations = [range(len(gadget1.locations) + len(rotated_gadget2.locations))] 

        # Now we have to combine transitions. In a euclidian type fashion.
    
        # If x->y is state 0 to 1 for gadget 1, and we have now k+z putting gadget 2 from stae 0 to 1, 
        # now we should have state (0,0), (0,1) etc


        pass

    def canonicalise():
        """
        The idea is to in some way describe how all the gadgets have been put together.
        """
        #TODO: Implement
        pass

