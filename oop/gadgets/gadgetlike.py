"""
Author: Zac Nwogwugwu 2025 (Updated version)

Handling the logic for gadget networks.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Set

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
    def getTransitions(self) -> Dict[int, List[Tuple[int, int, int]]]:
        """
        Returns transitions in a canonical format.
        e.g.
        {
        0 : [(0,1,1), (3,2,2)]
        }

        The above represents transitions in state 0:
        - From location 0 to location 1, changing the gadget state to 1
        - From location 3 to location 2, changing the gadget state to 2
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
        """
        Attempt to traverse the gadget from in_location to out_location.
        Returns True if the traverse was successful, False otherwise.
        Also updates the current state if successful.
        """
        state_transitions = self.transitions.get(self.current_state, [])
        for loc1, loc2, next_state in state_transitions:
            if loc1 == in_location and loc2 == out_location:
                self.current_state = next_state
                return True
        return False

    def getCurrentState(self):
        """Get the current state of the gadget, ensuring it's always an integer"""
        if isinstance(self.current_state, tuple):
            return self.current_state[0]
        return int(self.current_state)
    
    def setCurrentState(self, state):
        """Set the current state of the gadget, ensuring it's always an integer"""
        states = self.getStates()
        if isinstance(state, tuple):
            state = state[0]
        state = int(state)
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
    
    def __eq__(self, other):
        """
        Compare gadgets for behavioral equivalence using list-based Hopcroft minimization
        followed by isomorphism testing.
        """
        # Quick rejection tests
        states1 = self.getStates()
        states2 = other.getStates()
        locs1 = self.getLocations()
        locs2 = other.getLocations()
        
        if not states1 or not states2:
            return False
        
        if len(locs1) != len(locs2):
            return False
        
        try:
            # Get reachable parts of both gadgets (important for combined gadgets)
            reachable_self = self._get_reachable_gadget()
            reachable_other = other._get_reachable_gadget()
            
            # Prepare DFAs for minimization (no need to convert to dict format)
            dfa1 = (
                reachable_self.getStates(), 
                reachable_self.getLocations(), 
                reachable_self.getTransitions(), 
                reachable_self.current_state, 
                reachable_self.getStates()[1:] if len(reachable_self.getStates()) > 1 else []
            )
            
            dfa2 = (
                reachable_other.getStates(), 
                reachable_other.getLocations(), 
                reachable_other.getTransitions(), 
                reachable_other.current_state, 
                reachable_other.getStates()[1:] if len(reachable_other.getStates()) > 1 else []
            )
            
            # Use your list-based Hopcroft minimization
            min_dfa1 = hp.list_hopcroft_minimisation(*dfa1)
            min_dfa2 = hp.list_hopcroft_minimisation(*dfa2)
            
            # Normalize location numbering for comparison
            norm_dfa1 = hp.list_normalisation(min_dfa1)
            norm_dfa2 = hp.list_normalisation(min_dfa2)
            
            # Check isomorphism between the minimized DFAs
            return self._are_dfas_isomorphic(norm_dfa1, norm_dfa2)
            
        except Exception as e:
            print(f"Error in gadget comparison: {str(e)}")
            print(f"Current state context:")
            print(f"States 1: {states1}")
            print(f"States 2: {states2}")
            return False

    def _get_reachable_gadget(self):
        """Extract only the reachable portion of a gadget from its current state."""
        transitions = self.getTransitions()
        
        # Find reachable states through BFS
        reachable = set([self.current_state])
        frontier = [self.current_state]
        
        while frontier:
            state = frontier.pop(0)
            for _, _, next_state in transitions.get(state, []):
                if next_state not in reachable:
                    reachable.add(next_state)
                    frontier.append(next_state)
        
        # Create filtered transitions dict with only reachable states
        reachable_transitions = {}
        for state in reachable:
            reachable_transitions[state] = transitions.get(state, [])
        
        # Create a new gadget with only reachable states
        return Gadget(
            name=f"Reachable({self.name})",
            locations=self.locations,
            states=list(reachable),
            transitions=reachable_transitions,
            current_state=self.current_state
        )

    def _are_dfas_isomorphic(self, dfa1, dfa2):
        """
        Check if two minimized DFAs are isomorphic, accounting for
        different location numbering.
        """
        (states1, locs1, trans1, start1, accept1) = dfa1
        (states2, locs2, trans2, start2, accept2) = dfa2
        
        # Basic structure checks
        if len(states1) != len(states2) or len(locs1) != len(locs2):
            return False
            
        if start1 != start2:
            return False
            
        if set(accept1) != set(accept2):
            return False
        
        # Try all possible mappings of locations
        import itertools
        
        for loc_perm in itertools.permutations(locs2, len(locs1)):
            # Create location mapping
            loc_map = {locs1[i]: loc_perm[i] for i in range(len(locs1))}
            
            # Check if this mapping makes the transitions equivalent
            if self._check_transition_equivalence(states1, trans1, trans2, loc_map):
                return True
                
        return False

    def _check_transition_equivalence(self, states, trans1, trans2, loc_map):
        """Check if transitions are equivalent under the given location mapping."""
        # Convert list-based transitions to dictionaries for easier comparison
        dict_trans1 = {}
        dict_trans2 = {}
        
        for state in states:
            dict_trans1[state] = {}
            dict_trans2[state] = {}
            
            # Convert transitions for the first DFA
            for loc1, loc2, next_state in trans1.get(state, []):
                dict_trans1[state][(loc1, loc2)] = next_state
                
            # Convert transitions for the second DFA
            for loc1, loc2, next_state in trans2.get(state, []):
                dict_trans2[state][(loc1, loc2)] = next_state
        
        # Check transition equivalence
        for state in states:
            # Map transitions from dfa1 to how they'd appear in dfa2
            mapped_trans = {}
            for (loc1_in, loc1_out), next_state in dict_trans1[state].items():
                mapped_loc_in = loc_map[loc1_in]
                mapped_loc_out = loc_map[loc1_out]
                mapped_trans[(mapped_loc_in, mapped_loc_out)] = next_state
            
            # Compare mapped transitions with dfa2's transitions
            if mapped_trans != dict_trans2[state]:
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

    def __iadd__(self, other):
        """
        To add a gadgetlike object to the gadget network, simply do X += y.
        """
        self.subgadgets.append(other)
        return self

    def connect(self, gadget, loc1, loc2):
        self.operations.append(("CONNECT", gadget, loc1, loc2))

    def combine(self, gadget1_index, gadget2_index, rotation, splicing_index=-1):
        self.operations.append(("COMBINE", gadget1_index, gadget2_index, rotation, splicing_index))

    def do_connect(self, gadget, loc1, loc2):
        """Connect two locations within a gadget, modifying the gadget in place."""
        if loc1 not in gadget.locations or loc2 not in gadget.locations:
            raise ValueError(f"One or both locations not in {gadget.name}")

        # Group transitions by their start and end locations
        inbound1 = {}  # Transitions ending at loc1
        outbound1 = {}  # Transitions starting at loc1
        inbound2 = {}  # Transitions ending at loc2
        outbound2 = {}  # Transitions starting at loc2
        
        # New transitions that will replace the old ones
        new_transitions = {}
        
        # Initialize new transitions with transitions that don't involve loc1 or loc2
        for state, transitions_list in gadget.transitions.items():
            new_transitions[state] = []
            
            for loc_in, loc_out, next_state in transitions_list:
                # Keep transitions that don't involve the locations we're connecting
                if loc_in != loc1 and loc_in != loc2 and loc_out != loc1 and loc_out != loc2:
                    new_transitions[state].append((loc_in, loc_out, next_state))
                
                # Categorize transitions involving loc1 and loc2
                if loc_in == loc1 and loc_out != loc2:
                    outbound1.setdefault(state, []).append((loc_out, next_state))
                elif loc_in == loc2 and loc_out != loc1:
                    outbound2.setdefault(state, []).append((loc_out, next_state))
                elif loc_out == loc1 and loc_in != loc2:
                    inbound1.setdefault(next_state, []).append((loc_in, state))
                elif loc_out == loc2 and loc_in != loc1:
                    inbound2.setdefault(next_state, []).append((loc_in, state))
        
        # Create new transitions through the connection
        for state in gadget.states:
            # Transitions: in -> loc1 -> loc2 -> out
            for (loc_in, in_state) in inbound1.get(state, []):
                for (loc_out, next_state) in outbound2.get(state, []):
                    new_transitions[in_state].append((loc_in, loc_out, next_state))
            
            # Transitions: in -> loc2 -> loc1 -> out
            for (loc_in, in_state) in inbound2.get(state, []):
                for (loc_out, next_state) in outbound1.get(state, []):
                    new_transitions[in_state].append((loc_in, loc_out, next_state))
        
        # Update the gadget
        gadget.transitions = new_transitions
        gadget.locations = [loc for loc in gadget.locations if loc != loc1 and loc != loc2]

    def do_combine(self, gadget1_index, gadget2_index, rotation, splicing_index=-1):
        gadget1 = self.subgadgets[gadget1_index]
        gadget2 = self.subgadgets[gadget2_index]

        modulo_index = len(gadget2.locations)
        rotated_locations = [((location + rotation) % modulo_index + splicing_index+1) for location in gadget2.locations]
        rotated_transitions = {
            i: [((l1 + rotation) % modulo_index + splicing_index+1, (l2 + rotation) % modulo_index + splicing_index+1, n) for l1, l2, n in d_i]
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
        
        gadget1transitions = {
            i: [(loc1 + modulo_index if loc1 > splicing_index else loc1, 
                loc2 + modulo_index if loc2 > splicing_index else loc2, 
                next_state) 
                for loc1, loc2, next_state in d_i] 
            for i, d_i in gadget1.transitions.items()
        }
        new_transitions = {}

        # TODO: need to generalise for arbitrary combinations. (i.e. d>2)
        for (s1, s2) in new_states:
            new_transitions[(s1, s2)] = []
            for locA, locB, next_state in gadget1transitions[s1]:
                new_transitions[(s1, s2)].append((locA, locB, (next_state, s2)))
            for locA, locB, next_state in rotated_transitions[s2]:
                new_transitions[(s1, s2)].append((locA, locB, (s1, next_state)))

        
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
        Process all operations to create a single simplified gadget.
        """
        if not self.subgadgets:
            return None
        
        combined = self.subgadgets[0]
        
        for op in self.operations:
            match op[0]:
                case "CONNECT":
                    _, gadget_index, l1, l2 = op
                    # Convert the gadget index to a gadget object
                    if isinstance(gadget_index, int):
                        gadget = self.subgadgets[gadget_index]
                    else:
                        gadget = gadget_index
                    self.do_connect(gadget, l1, l2)
                case "COMBINE":
                    _, g1, g2, rot, smth = op
                    combined = self.do_combine(g1, g2, rot, smth)
        
        # Remove states with no outgoing transitions
        keys_to_remove = []
        for k, v in combined.transitions.items():
            if not v:
                keys_to_remove.append(k)
                
        for k in keys_to_remove:
            del combined.transitions[k]
            combined.states.remove(k) 
        
        return combined
    
