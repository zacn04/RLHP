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
    
    def __eq__(self, other):
        """
        Compare gadgets for behavioral equivalence, accounting for possible location
        isomorphism and state equivalence.
        """
        # Quick preliminary checks
        all_states1 = self.getStates()
        all_states2 = other.getStates()
        
        if not all_states1 or not all_states2:
            return False
        
        if len(all_states1) != len(all_states2):
            return False
            
        # Get transitions and locations
        transitions1 = self.getTransitions()
        transitions2 = other.getTransitions()
        locs1 = self.getLocations()
        locs2 = other.getLocations()
        
        # If locations don't match in count, can't be equivalent
        if len(locs1) != len(locs2):
            return False
        
        # Try every possible mapping of locations to see if we can find an isomorphism
        import itertools
        
        # Generate all possible location mappings
        for loc_mapping in itertools.permutations(locs2, len(locs1)):
            # Create a dictionary mapping from locs1 to the current permutation of locs2
            location_map = {locs1[i]: loc_mapping[i] for i in range(len(locs1))}
            
            # Check if this mapping creates equivalent behavior
            if self._check_behavioral_equivalence(transitions1, transitions2, location_map, all_states1, all_states2):
                return True
                
        return False
        
    def _check_behavioral_equivalence(self, transitions1, transitions2, location_map, states1, states2):
        """
        Check if the gadgets have equivalent behavior under the given location mapping.
        """
        # Build a mapping between states based on transition behavior
        state_map = {}
        reverse_state_map = {}
        
        # For each state in gadget1, try to find an equivalent state in gadget2
        for s1 in states1:
            if s1 in state_map:
                continue
                
            # Quick check: count transitions for this state
            trans_count1 = len(transitions1.get(s1, []))
            
            # For each unmapped state in gadget2, check if behavior matches
            for s2 in states2:
                if s2 in reverse_state_map:
                    continue
                    
                # Compare transition count first (quick rejection)
                if trans_count1 != len(transitions2.get(s2, [])):
                    continue
                    
                # Create copies of state maps for this attempt
                temp_state_map = state_map.copy()
                temp_reverse_map = reverse_state_map.copy()
                
                # Try mapping s1 to s2
                temp_state_map[s1] = s2
                temp_reverse_map[s2] = s1
                
                if self._transitions_match(transitions1.get(s1, []), transitions2.get(s2, []), 
                                        location_map, temp_state_map, temp_reverse_map):
                    # Update the maps
                    state_map = temp_state_map
                    reverse_state_map = temp_reverse_map
                    break
            
        # If we couldn't map all states, the gadgets aren't equivalent
        return len(state_map) == len(states1)
        
    def _transitions_match(self, trans1, trans2, location_map, state_map, reverse_state_map):
        """
        Check if transitions match under the given mapping.
        """
        # If the number of transitions doesn't match, they can't be equivalent
        if len(trans1) != len(trans2):
            return False
            
        # Create copies to avoid modifying the originals
        state_map_copy = state_map.copy()
        reverse_state_map_copy = reverse_state_map.copy()
        
        # Map transitions from gadget1 to how they'd appear in gadget2
        mapped_trans1 = []
        for loc1_in, loc1_out, next_state1 in trans1:
            mapped_loc_in = location_map[loc1_in]
            mapped_loc_out = location_map[loc1_out]
            mapped_trans1.append((mapped_loc_in, mapped_loc_out, next_state1))
            
        # Check if all transitions in gadget2 have a match
        for loc2_in, loc2_out, next_state2 in trans2:
            # Try to find a matching transition
            found_match = False
            for mapped_in, mapped_out, next_state1 in mapped_trans1:
                if loc2_in == mapped_in and loc2_out == mapped_out:
                    # Check state compatibility
                    if next_state1 in state_map_copy:
                        if state_map_copy[next_state1] == next_state2:
                            found_match = True
                            break
                    elif next_state2 not in reverse_state_map_copy:
                        # Create new state mapping
                        state_map_copy[next_state1] = next_state2
                        reverse_state_map_copy[next_state2] = next_state1
                        found_match = True
                        break
                        
            if not found_match:
                return False
                
        # If all transitions match, update the state maps
        state_map.update(state_map_copy)
        reverse_state_map.update(reverse_state_map_copy)
        
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
        """Combine two gadgets into a new one with potentially rotated locations."""
        gadget1 = self.subgadgets[gadget1_index]
        gadget2 = self.subgadgets[gadget2_index]
        
        # Find the highest location number across all gadgets to avoid conflicts
        max_existing_loc = max(
            max(g.locations) if g.locations else 0 
            for g in self.subgadgets
        )
        offset = max_existing_loc + 1
        
        # Map original locations in gadget2 to new rotated locations
        old_to_new_locs = {}
        g2_locs = gadget2.getLocations()
        
        for i, loc in enumerate(g2_locs):
            new_loc = ((i + rotation) % len(g2_locs)) + offset
            old_to_new_locs[loc] = new_loc
        
        # Get rotated locations for gadget2
        rotated_locations = [old_to_new_locs[loc] for loc in g2_locs]
        
        # Rotate transitions in gadget2
        rotated_transitions = {}
        for state, transitions_list in gadget2.transitions.items():
            rotated_transitions[state] = []
            for loc1, loc2, next_state in transitions_list:
                new_loc1 = old_to_new_locs[loc1]
                new_loc2 = old_to_new_locs[loc2]
                rotated_transitions[state].append((new_loc1, new_loc2, next_state))
        
        # Combine locations
        new_locations = (
            gadget1.locations[:splicing_index+1] + 
            rotated_locations + 
            gadget1.locations[splicing_index+1:]
        )
        
        # Create compound states (product construction)
        new_states = [(s1, s2) for s1 in gadget1.states for s2 in gadget2.states]
        new_transitions = {}
        
        for (s1, s2) in new_states:
            new_transitions[(s1, s2)] = []
            
            # Add transitions from gadget1 (stay in same gadget2 state)
            for loc1, loc2, next_state in gadget1.transitions.get(s1, []):
                new_transitions[(s1, s2)].append((loc1, loc2, (next_state, s2)))
            
            # Add transitions from gadget2 (stay in same gadget1 state)
            for loc1, loc2, next_state in rotated_transitions.get(s2, []):
                new_transitions[(s1, s2)].append((loc1, loc2, (s1, next_state)))
        
        # Remove duplicate locations
        new_locations = list(dict.fromkeys(new_locations))
        
        # Create and return the combined gadget
        return Gadget(
            name=f"Combined({gadget1.name}+{gadget2.name})",
            locations=new_locations,
            states=new_states,
            transitions=new_transitions,
            current_state=(gadget1.current_state, gadget2.current_state)
        )

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