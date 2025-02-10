"""
Author: Zac Nwogwugwu 2025

Handling the logic for gadget networks.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

from oop.dfa import hopcroft as hp
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

    def __str__(self):
        return f"{self.name}"
    def traverse(self, in_location, out_location):
        #Generalised traversal logic
        try:
            self.current_state = self.transitions[self.current_state][(in_location, out_location)]
            return True
        except KeyError:
            return False
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

    
    def __eq__(self, other):

        all_states1 = self.getStates()
        dfa1 =  \
            all_states1, self.getLocations(), self.getTransitions(), all_states1[0], all_states1[1:]
        all_states2 = other.getStates()
        dfa2 =  \
            all_states2, other.getLocations(), other.getTransitions(), all_states2[0], all_states2[1:]
        
        
        minimised_dfa1 = hp.hopcroft_minimisation(*dfa1)
        minimised_dfa2 = hp.hopcroft_minimisation(*dfa2)

        minimised_dfa1 = hp.normalisation(minimised_dfa1)
        minimised_dfa2 = hp.normalisation(minimised_dfa2)

        (min_states1, locs1, min_transitions1, min_start1, min_accepting1) = minimised_dfa1
        (min_states2, locs2, min_transitions2, min_start2, min_accepting2) = minimised_dfa2


        # Ok, technically a DFA does not describe its locations, but this is harmless.

        if len(min_states1) != len(min_states2):
            return False
    
        if min_start1 != min_start2:
            return False
        
        if set(min_accepting1) != set(min_accepting2):
            return False
        
        for state in min_states1:
            for loc1 in locs1:  # Locations (alphabet)
                for loc2 in locs2:
                    next_state1 = min_transitions1[state].get((loc1, loc2), -1)
                    next_state2 = min_transitions2[state].get((loc1, loc2), -1)
                    if next_state1 != next_state2:
                        return False
                    
        return True
        
class GadgetNetwork(GadgetLike):
    def __init__(self, name="GadgetNetwork"):
        """
        The environment in which gadgets will be manipulated.
        """
        self.name = name
        self.subgadgets = []
        self.operations = []

    def __str__(self):
        res = ""
        for i,s in enumerate(self.subgadgets):
            res += f"{i}: {s} \n"
        return res


    #TODO: find a way to index the subgadgets nicely..

    def __iadd__(self, other):
        """
        To add a gadgetlike object to the gadget network, simply do X += y.
        """
        self.subgadgets.append(other)
        return self

    def connect(self, gadget, loc1, loc2):
        self.operations.append(("CONNECT", gadget, loc1, loc2))

    def combine(self, gadget1_index,gadget2_index, rotation, splicing_index=-1):
        self.operations.append(("COMBINE", gadget1_index, gadget2_index, rotation, splicing_index))

    def do_connect(self, gadget, loc1, loc2):
        #Assuming that we can only connect within the same gadget. We modify this gadget in place.
        if loc1 not in gadget.locations or loc2 not in gadget.locations:
            raise ValueError(f"One or both locations not in {gadget.name}")

        new_transitions = {}
        
        inbound1 = {}
        outbound1 = {}
        inbound2 = {}
        outbound2 = {}

        for state, loc_map in gadget.transitions.items():
            new_loc_map = {}
            for (locA, locB), next_state in loc_map.items():
                
                # Replace loc2 with loc1 in transitions
                if locA != loc2 and locA != loc1 and locB != loc2 and locB != loc1:
                    new_loc_map[(locA, locB)] = next_state

                if locA == loc2 and locB != loc1:
                    outbound2.setdefault(state, []).append((locB, next_state))

                if locA == loc1 and locB != loc2:
                    outbound1.setdefault(state, []).append((locB, next_state))
                
                if locB == loc2 and locA != loc1:
                    inbound2.setdefault(next_state, []).append((locA, state))

                if locB == loc1 and locA != loc2:
                    inbound1.setdefault(next_state, []).append((locA, state))
                
            new_transitions[state] = new_loc_map


        for state, loc_map in gadget.transitions.items():
            for (locA, in_state) in inbound1.get(state, []):
                for (locB, next_state) in outbound2.get(state, []):
                    new_transitions[in_state][(locA, locB)] = next_state
            for (locA, in_state) in inbound2.get(state, []):
                for (locB, next_state) in outbound1.get(state, []):
                    new_transitions[in_state][(locA, locB)] = next_state


        gadget.transitions = new_transitions
        gadget.locations = [loc for loc in gadget.locations if loc != loc2 and loc != loc1]




    def do_combine(self, gadget1_index, gadget2_index, rotation, splicing_index=-1):

        gadget1 = self.subgadgets[gadget1_index]
        gadget2 = self.subgadgets[gadget2_index]

        modulo_index = len(gadget2.locations)
        rotated_locations = [((location + rotation) % modulo_index + splicing_index+1) for location in gadget2.locations]
        rotated_transitions = {
            i: {((k[0] + rotation) % modulo_index + splicing_index+1, (k[1] + rotation) % modulo_index + splicing_index+1): v for k, v in d_i.items()}
            for i, d_i in gadget2.transitions.items()
        }

        #renames the locations
        rotated_gadget2 = Gadget(
            name=f"{rotation}-Rotated {gadget2.name}",
            locations=rotated_locations,
            states=gadget2.states, #assuming the rotations have no bearings of states
            transitions=rotated_transitions,
            current_state=gadget2.current_state
            )
        

        new_locations = gadget1.locations[:splicing_index+1] + rotated_gadget2.locations + [location+modulo_index for location in gadget1.locations[splicing_index+1:]]

        #after this, we have both transitions, separate. need to combine and not lose any lol
        new_states = [(s1, s2) for s1 in gadget1.states for s2 in rotated_gadget2.states]
        
        gadget1transitions = {i: {tuple(x + modulo_index if x > splicing_index else x for x in key): value for key, value in d_i.items()}
        for i, d_i in gadget1.transitions.items()}

        new_transitions = {}

        for (s1, s2) in new_states:
            new_transitions[(s1, s2)] = {}

            for (locA, locB), next_state in gadget1transitions[s1].items():
                new_transitions[(s1, s2)][(locA, locB)] = (next_state, s2)

            for (locA, locB), next_state in rotated_transitions[s2].items():
                new_transitions[(s1, s2)][(locA, locB)] = (s1, next_state)

        
        new_gadget = Gadget(
        name=f"Combined({gadget1.name}+{gadget2.name})",
        locations=new_locations,
        states=new_states,
        transitions=new_transitions,
        current_state=(gadget1.current_state, gadget2.current_state)
        )

        return new_gadget

    def simplify(self):
        """
        The idea is to in some way describe how all the gadgets have been put together.
        """
        if not self.subgadgets:
            return None
        combined = self.subgadgets[0]

        for op in self.operations:
            match op[0]:
                case "CONNECT":
                    _, gadget, l1, l2 = op
                    self.do_connect(combined, l1, l2)
                case "COMBINE":
                    _, g1, g2, rot, smth = op
                    combined = self.do_combine(g1, g2, rot, smth)
        
        keys_to_remove = [k for k, v in combined.transitions.items() if not v]
        for k in keys_to_remove:
            del combined.transitions[k]
            combined.states.remove(k) 

        return combined