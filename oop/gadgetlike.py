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
        #Gadget 1 and Gadget 2 can be equal.
        self.connections.append(("CONNECT", gadget1, gadget2, loc1, loc2))
        #having this will mean that the canonicalisation will have these gadgets connected to one another.
        #Or we could just have that the locations are conncted.

    def combine(self, gadget1, gadget2, rotation, splicing_index=-1):
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
        
        #TODO: implement correct splicing index logic
        offset = len(gadget1.locations) #tacking on things to the end, ignoring splciing index for now. 

        offset_rotated_locations = [location + offset for location in rotated_gadget2.locations]

        offset_rotated_transitions = offset_transitions(rotated_gadget2.transitions, offset)

        offset_transitions = lambda transitions, offset: {
        state: {loc + offset: nxt for loc, nxt in inner_dict.items()}
        for state, inner_dict in transitions.items()
        }

        new_locations = list(gadget1.locations) + offset_rotated_locations

        new_states = []
        for s1 in gadget1.states:
            for s2 in rotated_gadget2.states:
                new_states.append((s1, s2))
        new_transitions = {}

        # Now we have to combine transitions. In a euclidian type fashion.
    
        # If x->y is state 0 to 1 for gadget 1, and we have now k+z putting gadget 2 from stae 0 to 1, 
        # now we should have state (0,0), (0,1) etc

        for state_1 in gadget1.states:
            for state_2 in rotated_gadget2.states:

                
                if s1 in gadget1.transitions:
                    for loc1, next_s1 in gadget1.transitions[s1].items():
                        new_transitions[(s1, s2)][loc1] = (next_s1, s2)

                if s2 in gadget1.transitions:
                    for loc1, next_s2 in offset_rotated_transitions[s2].items():
                        new_transitions[(s1, s2)][loc1] = (s1, next_s2)

            
        new_gadget = Gadget(
        name=f"Combined({gadget1.name}+{gadget2.name})",
        locations=new_locations,
        states=new_states,
        transitions=new_transitions,
        current_state=(gadget1.current_state, gadget2.current_state)
        )



        return new_gadget

    def canonicalise():
        """
        The idea is to in some way describe how all the gadgets have been put together.
        """
        #TODO: Implement
        pass

