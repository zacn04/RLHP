"""
Author: Zac Nwogwugwu 2025

Handling the logic for gadget networks.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

class GadgetLike:
    """
    The base class for gadgets/gadget networks.
    """
    @abstractmethod
    def getLocations(self) -> List[int]:
        pass

    @abstractmethod
    def getStates(self) -> List[int]:
        pass

    @abstractmethod
    def getTransitions(self) -> Dict[int, Dict[Tuple[int, int], int]]:
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

class Gadget(GadgetLike):
    def __init__(self, name=None, locations=None, states=None, transitions=None, current_state=0):
        self.locations = locations
        self.states = states
        self.name = name
        self.transitions = transitions
        self.current_state = current_state

    def __repr__(self):
        return f"{self.name}"
    
    def traverse(self, in_location, out_location):
        #Generalised traversal logic
        try:
            self.current_state = self.transitions[self.current_state][(in_location, out_location)]
            return True
        except KeyError:
            return False
            '''raise TraversalError(f"Invalid transition from {in_location} to {out_location} in state {self.current_state}")
        finally:
            pass'''

    def getCurrentState(self):
        return self.current_state
    
    def setCurrentState(self, state):
        states = self.getStates()
        if state in states:
            self.current_state = state
        else:
            raise ValueError("State not found")
    
    def getLocations(self):
        return self.locations
    
    def getStates(self):
        return self.states
    
    def getTransitions(self):
        return self.transitions
    
    def __getitem__(self, transition): #allows for easier transitions.
        self.traverse(transition[0], transition[1])

class GadgetNetwork(GadgetLike):
    def __init__(self, name="GadgetNetwork"):
        """
        The environment in which gadgets will be manipulated.
        """
        self.name = name
        self.subgadgets = []
        self.operations = []


    #MAYBE find a way to index the subgadgets nicely..

    def __iadd__(self, other):
        """
        To add a gadgetlike object to the gadget network, simply do X += y.
        """
        self.subgadgets.append(other)
        return self

    def connect(self, gadget, loc1, loc2):
        self.operations.append(("CONNECT", gadget, loc1, loc2))

    def combine(self, gadget1,gadget2, rotation, splicing_index=-1):
        self.operations.append(("COMBINE", gadget1, gadget2, rotation))

    def do_connect(self, gadget, loc1, loc2):
        '''#Gadget 1 and Gadget 2 can be equal.
        self.connections.append(("CONNECT", gadget1, gadget2, loc1, loc2))
        #having this will mean that the canonicalisation will have these gadgets connected to one another.
        #Or we could just have that the locations are conncted.'''

        #Assuming that we can only connect within the same gadget. We modify this gadget in place.

        if loc1 not in gadget.locations or loc2 not in gadget.locations:
            raise ValueError(f"One or both locations not in {gadget.name}")

        new_transitions = {}
        for state, loc_map in gadget.transitions.items():
            new_loc_map = {}
            for old_transition, next_state in loc_map.items():
                locA, locB = old_transition
                newA = loc1 if locA == loc2 else locA
                newB = loc1 if locB == loc2 else locB
                if (newA, newB) in new_loc_map:
                    if new_loc_map[(newA, newB)] != next_state:
                        raise ValueError(f"Conflict in transitions for merged location {(newA, newB)}")
                else:
                    new_loc_map[(newA, newB)] = next_state
            new_transitions[state] = new_loc_map

        

        gadget.transitions = new_transitions
        if loc2 in gadget.locations:
            gadget.locations.remove(loc2)

        #self.operations.append(("CONNECT", gadget, loc1, loc2))



    def do_combine(self, gadget1, gadget2, rotation, splicing_index=-1):
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

        offset_transitions = lambda transitions, offset: {
        state: {(locA + offset, locB + offset): nxt for (locA, locB), nxt in inner_dict.items()}
        for state, inner_dict in transitions.items()
        }

        offset_rotated_transitions = offset_transitions(rotated_gadget2.transitions, offset)

        

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

                new_transitions[(state_1, state_2)] = {}
                
                if state_1 in gadget1.transitions:
                    for loc1, next_s1 in gadget1.transitions[s1].items():
                        new_transitions[(state_1, state_2)][loc1] = (next_s1, state_2)

                if state_2 in offset_rotated_transitions:
                    for loc2, next_s2 in offset_rotated_transitions[s2].items():
                        new_transitions[(state_1, state_2)][loc2] = (state_1, next_s2)

        for op in self.operations:
            if op[0] == "CONNECT":
                _, _, loc1, loc2 = op
                if loc1 in gadget1.locations and loc2 in rotated_gadget2.locations:
                    bridge_loc2 = loc2 + offset  
                    for state_1 in gadget1.states:
                        for state_2 in rotated_gadget2.states:
                            combined_state = (state_1, state_2)
                            if (loc1, bridge_loc2) not in new_transitions[combined_state]:
                                new_transitions[combined_state][(loc1, bridge_loc2)] = (state_1, state_2)

            
        new_gadget = Gadget(
        name=f"Combined({gadget1.name}+{gadget2.name})",
        locations=new_locations,
        states=new_states,
        transitions=new_transitions,
        current_state=(gadget1.current_state, gadget2.current_state)
        )


        #self.operations.append("COMBINE", gadget1, gadget2, rotation)
        return new_gadget

    def canonicalise(self):
        """
        The idea is to in some way describe how all the gadgets have been put together.
        """
        if not self.subgadgets:
            return None
        
        combined = self.subgadgets[0]

        for op in self.operations:
            match op[0]:
                case "CONNECT":
                    _, _, l1, l2 = op
                    self.do_connect(combined, l1, l2)
                case "COMBINE":
                    _, g1, g2, rot = op
                    combined = self.do_combine(g1, g2, rot)

        return combined
                    


        

