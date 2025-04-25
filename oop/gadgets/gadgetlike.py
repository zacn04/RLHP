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
    
    def simplify_gadget(gadget):
        """
        Simplify a gadget by removing states with no outgoing transitions.
        This is particularly useful for combined gadgets which may have empty states.
        
        Args:
            gadget: The gadget to simplify
            
        Returns:
            A new gadget with empty states removed
        """
        # Get all states with outgoing transitions
        non_empty_states = []
        filtered_transitions = {}
        
        for state, transitions in gadget.transitions.items():
            if transitions:  # If this state has any transitions
                filtered_transitions[state] = transitions
                non_empty_states.append(state)
        
        # Check if the current state is in the filtered states
        # If not, use the first available state as current
        if gadget.current_state in non_empty_states:
            new_current_state = gadget.current_state
        elif non_empty_states:
            new_current_state = non_empty_states[0]
        else:
            # No states with transitions, keep original current state
            new_current_state = gadget.current_state
        
        # Create a new simplified gadget
        simplified = Gadget(
            name=gadget.name,
            locations=gadget.locations,
            states=non_empty_states,
            transitions=filtered_transitions,
            current_state=new_current_state
        )
        
        return simplified

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
        Compare gadgets for structural isomorphism with detailed debugging.
        """
        print("\n----- STARTING COMPARISON -----")
        print(f"Comparing gadget1: {self.name} with gadget2: {other.name}")
        
        # Quick rejection tests
        if not isinstance(other, Gadget):
            print("Rejection: other is not a Gadget instance")
            return False
                
        states1 = self.getStates()
        states2 = other.getStates()
        locs1 = self.getLocations()
        locs2 = other.getLocations()
        
        print(f"Gadget1 states: {states1}")
        print(f"Gadget2 states: {states2}")
        print(f"Gadget1 locations: {locs1}")
        print(f"Gadget2 locations: {locs2}")
        
        # Verify basic structural properties
        if len(states1) != len(states2):
            print(f"Rejection: Number of states doesn't match: {len(states1)} vs {len(states2)}")
            return False
        
        if len(locs1) != len(locs2):
            print(f"Rejection: Number of locations doesn't match: {len(locs1)} vs {len(locs2)}")
            return False
        
        # Reject trivial case where untransformed gadgets of the same class are compared
        if (self.__class__.__name__ == other.__class__.__name__ and 
            self.name == other.name and 
            not "_" in self.name and 
            not "Combined" in self.name):
            print("Rejection: Same class and name without modifications")
            return False
        
        # Get transition structures for both gadgets
        transitions1 = self.getTransitions()
        transitions2 = other.getTransitions()
        
        print("\nTransitions for gadget1:")
        for state, trans in transitions1.items():
            print(f"  State {state}: {trans}")
        
        print("\nTransitions for gadget2:")
        for state, trans in transitions2.items():
            print(f"  State {state}: {trans}")
        
        # Ensure all states have entries in transitions (even if empty)
        for state in states1:
            if state not in transitions1:
                transitions1[state] = []
        for state in states2:
            if state not in transitions2:
                transitions2[state] = []
        
        # Try all possible state mappings
        import itertools
        state_permutations = list(itertools.permutations(states2, len(states1)))
        print(f"\nTrying {len(state_permutations)} possible state mappings")
        
        for perm_idx, state_mapping in enumerate(state_permutations):
            state_map = {s1: s2 for s1, s2 in zip(states1, state_mapping)}
            print(f"\nTesting state mapping #{perm_idx+1}: {state_map}")
            
            # Try all possible cyclic shifts of locations
            n_locs = len(locs1)
            print(f"Trying {n_locs} forward cyclic shifts")
            
            for shift in range(n_locs):
                # Create a cyclic shift mapping from gadget1 locations to gadget2 locations
                loc_map = {}
                for i, loc1 in enumerate(locs1):
                    # Apply cyclic shift to get corresponding location in gadget2
                    shifted_idx = (i + shift) % n_locs
                    loc_map[loc1] = locs2[shifted_idx]
                
                print(f"  Testing shift {shift}: {loc_map}")
                
                # Check if this mapping makes transitions from gadget1 match those in gadget2
                equiv = self._check_transition_equivalence_debug(transitions1, transitions2, state_map, loc_map)
                if equiv:
                    print(f">>> MATCH FOUND! With shift {shift} and state mapping {state_map}")
                    return True
            
            # Also try reversed cyclic ordering (for gadgets that could be mirror images)
            print(f"Trying {n_locs} reversed cyclic shifts")
            
            for shift in range(n_locs):
                # Create a reversed cyclic shift mapping
                loc_map = {}
                for i, loc1 in enumerate(locs1):
                    # Apply reversed cyclic shift
                    shifted_idx = (shift - i) % n_locs
                    loc_map[loc1] = locs2[shifted_idx]
                
                print(f"  Testing reversed shift {shift}: {loc_map}")
                
                # Check if this mapping makes transitions from gadget1 match those in gadget2
                equiv = self._check_transition_equivalence_debug(transitions1, transitions2, state_map, loc_map)
                if equiv:
                    print(f">>> MATCH FOUND! With reversed shift {shift} and state mapping {state_map}")
                    return True
        
        # No isomorphism found
        print("No isomorphism found after trying all mappings")
        return False

    def _check_transition_equivalence_debug(self, transitions1, transitions2, state_map, loc_map):
        """
        Check if transitions are equivalent under given state and location mappings.
        With detailed debugging output.
        """
        for state1, trans_list1 in transitions1.items():
            state2 = state_map[state1]
            trans_list2 = transitions2.get(state2, [])
            
            print(f"    Checking state {state1} → {state2}")
            print(f"      Original transitions: {trans_list1}")
            print(f"      Target transitions: {trans_list2}")
            
            # Map transitions from gadget1 to their expected form in gadget2
            mapped_trans1 = []
            for from_loc1, to_loc1, next_state1 in trans_list1:
                from_loc2 = loc_map[from_loc1]
                to_loc2 = loc_map[to_loc1]
                next_state2 = state_map[next_state1]
                mapped_trans = (from_loc2, to_loc2, next_state2)
                mapped_trans1.append(mapped_trans)
                print(f"      Mapping: ({from_loc1},{to_loc1},{next_state1}) → {mapped_trans}")
            
            # Convert both to sets for comparison
            mapped_trans1_set = set(mapped_trans1)
            trans_list2_set = set(trans_list2)
            
            print(f"      Mapped transitions set: {mapped_trans1_set}")
            print(f"      Target transitions set: {trans_list2_set}")
            print(f"      Sets match: {mapped_trans1_set == trans_list2_set}")
            
            # Check if they match
            if mapped_trans1_set != trans_list2_set:
                print(f"      Transitions don't match for state {state1} → {state2}")
                return False
        
        print("    All transitions match!")
        return True

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

    
    def _check_full_transition_equivalence(self, transitions1, transitions2, state_map, loc_map):
        """Check if transitions are equivalent under the given mappings."""
        # Check transitions for each state in this gadget
        for state1, trans_list1 in transitions1.items():
            state2 = state_map[state1]
            trans_list2 = transitions2.get(state2, [])
            
            # Convert transitions to sets for comparison
            mapped_trans1 = {(loc_map[from_loc], loc_map[to_loc], state_map[next_state]) 
                            for from_loc, to_loc, next_state in trans_list1}
            
            trans2_set = {(from_loc, to_loc, next_state) 
                        for from_loc, to_loc, next_state in trans_list2}
            
            if mapped_trans1 != trans2_set:
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

        # # Initialize new transitions with transitions that don't involve loc1 or loc2
        # for state, transitions_list in gadget.transitions.items():
        #     new_transitions[state] = []
            
        #     for loc_in, loc_out, next_state in transitions_list:
        #         # Keep transitions that don't involve the locations we're connecting
        #         if loc_in != loc2 and loc_out != loc2:
        #             new_transitions[state].append((loc_in, loc_out, next_state))
                
        #         # Categorize transitions involving loc1 and loc2
        #         elif loc_in == loc2:
        #             new_transitions[state].append((loc1, loc_out, next_state))
        #         elif loc_out == loc2 and loc_in != loc2:
        #             new_transitions[state].append((loc_in, loc1, next_state))
                
        
        # Update the gadget
        gadget.transitions = new_transitions
        gadget.locations = [loc for loc in gadget.locations if loc != loc2 and loc != loc1]

        gadget=gadget.simplify_gadget()

    def do_combine(self, gadget1_index, gadget2_index, rotation, reflect = False, splicing_index=-1):
        """
        Combine two gadgets into a single gadget.
        
        Args:
            gadget1_index: Index of the left gadget
            gadget2_index: Index of the right gadget
            rotation: Rotation to apply to the right gadget's locations
            splicing_index: Index in the left gadget's location sequence after which to splice the right gadget
        
        Returns:
            A new combined gadget
        """
        gadget1 = self.subgadgets[gadget1_index]  # Left gadget
        gadget2 = self.subgadgets[gadget2_index]  # Right gadget
        
        # Step 1: Apply rotation to the right gadget's locations
        # This creates a cyclic shift of the location indices
        modulo_index = len(gadget2.locations)
        rotated_locations = [(location + rotation) % modulo_index for location in gadget2.locations]
        
        # Create rotated transitions for gadget2
        rotated_transitions = {}
        for state, transitions_list in gadget2.transitions.items():
            rotated_transitions[state] = []
            for loc1, loc2, next_state in transitions_list:
                # Apply the same rotation to transition locations
                rotated_loc1 = (loc1 + rotation) % modulo_index
                rotated_loc2 = (loc2 + rotation) % modulo_index
                rotated_transitions[state].append((rotated_loc1, rotated_loc2, next_state))
        
        # Step 2: Prepare for splicing by shifting indices
        # First, create a mapping for the rotated gadget2 locations
        # We need to shift all locations after the splice point in gadget1
        g1_locations = gadget1.locations
        g2_loc_count = len(rotated_locations)
        
        # Create a new location list for the combined gadget
        # First include locations from gadget1 up to the splice point
        new_locations = g1_locations[:splicing_index+1].copy()
        
        # Then include all rotated locations from gadget2
        # But we need to renumber them to avoid conflicts
        # Start numbering from the next available index
        g2_location_map = {}  # Maps original rotated locations to new indices
        for i, loc in enumerate(rotated_locations):
            new_loc = len(new_locations)
            g2_location_map[loc] = new_loc
            new_locations.append(new_loc)
        
        # Finally, include the remaining locations from gadget1
        # with appropriate index shifting
        g1_shift_map = {}  # Maps original gadget1 locations to new indices
        for i, loc in enumerate(g1_locations):
            if i <= splicing_index:
                # Locations before the splice point keep their indices
                g1_shift_map[loc] = loc
            else:
                # Locations after the splice point are shifted by the number of gadget2 locations
                new_loc = loc + g2_loc_count
                g1_shift_map[loc] = new_loc
                new_locations.append(new_loc)
        
        # Step 3: Create the new transitions for the combined gadget
        # States in the combined gadget are pairs (s1, s2) where s1 is from gadget1 and s2 from gadget2
        new_states = [(s1, s2) for s1 in gadget1.states for s2 in gadget2.states]
        new_transitions = {}
        
        # Initialize transitions for all state pairs
        for state_pair in new_states:
            new_transitions[state_pair] = []
        
        # Add transitions from gadget1, with remapped locations
        for s1 in gadget1.states:
            for s2 in gadget2.states:
                for loc1, loc2, next_state in gadget1.transitions.get(s1, []):
                    # Map the locations using the shift map
                    new_loc1 = g1_shift_map[loc1]
                    new_loc2 = g1_shift_map[loc2]
                    # The transition affects only the gadget1 state
                    new_transitions[(s1, s2)].append((new_loc1, new_loc2, (next_state, s2)))
        
        # Add transitions from gadget2, with remapped locations
        for s2 in gadget2.states:
            for s1 in gadget1.states:
                for loc1, loc2, next_state in rotated_transitions.get(s2, []):
                    # Map the locations using the gadget2 location map
                    new_loc1 = g2_location_map[loc1]
                    new_loc2 = g2_location_map[loc2]
                    # The transition affects only the gadget2 state
                    new_transitions[(s1, s2)].append((new_loc1, new_loc2, (s1, next_state)))
        
        # Remove states with no outgoing transitions from new_transitions
        non_empty_states = []
        filtered_transitions = {}

        for state, transitions in new_transitions.items():
            if transitions:  # If this state has any transitions
                filtered_transitions[state] = transitions
                non_empty_states.append(state)

        # Update new_transitions and new_states
        new_transitions = filtered_transitions
        new_states = non_empty_states

        
        new_locations.sort(reverse=reflect)


        # Create the new combined gadget with only non-empty states
        new_gadget = Gadget(
            name=f"Combined({gadget1.name}+{gadget2.name})",
            locations=new_locations,
            states=new_states,
            transitions=new_transitions,
            current_state=(gadget1.current_state, gadget2.current_state)
        )
        
        return new_gadget.simplify_gadget()
    
    def simplify(self):
        """
        Process all operations to create a single simplified gadget.
        Handles complex combinations of operations correctly.
        """
        if not self.subgadgets:
            return None
        
        # Make a copy of the subgadgets to work with
        working_gadgets = self.subgadgets.copy()
        
        # Process operations in order
        for op_idx, op in enumerate(self.operations):
            try:
                print(f"Processing operation {op_idx+1}/{len(self.operations)}: {op}")
                
                if op[0] == "CONNECT":
                    _, gadget_index, l1, l2 = op
                    
                    # Handle the gadget index correctly
                    if isinstance(gadget_index, int):
                        if gadget_index >= len(working_gadgets):
                            print(f"Warning: Gadget index {gadget_index} out of range (only {len(working_gadgets)} gadgets)")
                            continue
                        gadget = working_gadgets[gadget_index]
                    else:
                        gadget = gadget_index
                    
                    # Apply the connection
                    print(f"  Connecting locations {l1} and {l2} in gadget {gadget.name}")
                    print(f"  Gadget locations before: {gadget.locations}")
                    self.do_connect(gadget, l1, l2)
                    print(f"  Gadget locations after: {gadget.locations}")
                    
                elif op[0] == "COMBINE":
                    _, g1_idx, g2_idx, rotation, splice_idx, reflect = op
                    
                    # Validate indices
                    if g1_idx >= len(working_gadgets) or g2_idx >= len(working_gadgets):
                        print(f"Warning: Gadget indices ({g1_idx}, {g2_idx}) out of range (only {len(working_gadgets)} gadgets)")
                        continue
                    
                    # Get the gadgets to combine
                    gadget1 = working_gadgets[g1_idx]
                    gadget2 = working_gadgets[g2_idx]
                    
                    print(f"  Combining gadget {g1_idx} ({gadget1.name}) with gadget {g2_idx} ({gadget2.name})")
                    print(f"  Gadget1 locations: {gadget1.locations}")
                    print(f"  Gadget2 locations: {gadget2.locations}")
                    
                    # Create the combined gadget
                    result = self.do_combine(g1_idx, g2_idx, rotation, splice_idx, reflect)
                    
                    # Update the working gadgets list
                    # Remove gadgets in reverse order to avoid index shifting
                    if g2_idx > g1_idx:
                        working_gadgets.pop(g2_idx)
                        working_gadgets.pop(g1_idx)
                    else:
                        working_gadgets.pop(g1_idx)
                        working_gadgets.pop(g2_idx)
                    
                    # Add the combined gadget
                    working_gadgets.append(result)
                    
                    print(f"  Combined gadget locations: {result.locations}")
                    print(f"  Working gadgets count: {len(working_gadgets)}")
            
            except Exception as e:
                print(f"Error processing operation {op}: {str(e)}")
                print(f"Current working gadgets: {len(working_gadgets)}")
                for i, g in enumerate(working_gadgets):
                    print(f"  Gadget {i}: {g.name} with locations {g.locations}")
                raise  # Re-raise the exception to see the full traceback
        
        # After all operations, check if we have a single gadget
        if len(working_gadgets) != 1:
            print(f"Warning: After all operations, we have {len(working_gadgets)} gadgets instead of 1")
            # Return the first gadget as a fallback
            if working_gadgets:
                return working_gadgets[0]
            return None
        
        # We have a single gadget
        combined = working_gadgets[0]
        
        # Clean up states with no outgoing transitions
        keys_to_remove = []
        for k, v in combined.transitions.items():
            if not v:
                keys_to_remove.append(k)
                
        for k in keys_to_remove:
            del combined.transitions[k]
            if k in combined.states:  # Check if k is in states before removing
                combined.states.remove(k)
        
        return combined
        