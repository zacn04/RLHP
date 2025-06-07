"""
Author: Zac Nwogwugwu 2025 
Handling the logic for gadget networks
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from itertools import permutations
from oop.dfa import hopcroft as hp

class GadgetLike(ABC):
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
        Returns transitions in canonical format.
        """
        pass

class Gadget(GadgetLike):
    def __init__(self, name=None, locations=None, states=None, transitions=None, current_state=0):
        self.name = name
        self.locations = list(locations) if locations is not None else []
        self.states = list(states) if states is not None else []
        self.transitions = dict(transitions) if transitions is not None else {}
        self.current_state = current_state
        self.free_ports = set(locations)

    def __str__(self):
        lines = [
            f"Gadget {self.name}:",
            f"  Locations     : {self.getLocations()}",
            f"  States        : {self.getStates()}",
            f"  Current state : {self.getCurrentState()}",
            f"  Transitions   :"
        ]
        for state, trans_list in self.getTransitions().items():
            for inp, outp, nxt in trans_list:
                lines.append(f"    {state} --({inp}→{outp})-> {nxt}")
        return "\n".join(lines)

    def traverse(self, in_location, out_location):
        state_transitions = self.transitions.get(self.current_state, [])
        for loc1, loc2, next_state in state_transitions:
            if loc1 == in_location and loc2 == out_location:
                self.current_state = next_state
                return True
        return False

    def getCurrentState(self):
        if isinstance(self.current_state, tuple):
            return int(self.current_state[0])
        return int(self.current_state)

    def setCurrentState(self, state):
        val = state[0] if isinstance(state, tuple) else state
        val = int(val)
        if val in self.getStates():
            self.current_state = val
        else:
            raise ValueError("State not found")

    def getLocations(self):
        return list(self.locations)

    def getStates(self):
        return list(self.states)

    def getTransitions(self):
        return dict(self.transitions)
    
    def removePorts(self, *ports):
        for p in ports:
            self.free_ports.discard(p)

    def __eq__(self, other):
        # Quick shape checks
        if len(self.getLocations()) != len(other.getLocations()):
            return False
        # Extract reachable subgadgets
        r1 = self._get_reachable_gadget()
        r2 = other._get_reachable_gadget()
        # Build DFAs
        dfa1 = (
            r1.getStates(),
            r1.getLocations(),
            r1.getTransitions(),
            r1.current_state,
            [r1.current_state]
        )
        dfa2 = (
            r2.getStates(),
            r2.getLocations(),
            r2.getTransitions(),
            r2.current_state,
            [r2.current_state]
        )
        min1 = hp.list_hopcroft_minimisation(*dfa1)
        min2 = hp.list_hopcroft_minimisation(*dfa2)
        norm1 = hp.list_normalisation(min1)
        norm2 = hp.list_normalisation(min2)
        return self._are_dfas_isomorphic(norm1, norm2)

    def _get_reachable_gadget(self):
        trans = self.getTransitions()
        reachable = {self.current_state}
        frontier = [self.current_state]
        while frontier:
            s = frontier.pop()
            for _, _, ns in trans.get(s, []):
                if ns not in reachable:
                    reachable.add(ns)
                    frontier.append(ns)
        filtered = {s: trans.get(s, []) for s in reachable}
        return Gadget(
            name=f"Reachable({self.name})",
            locations=self.locations,
            states=list(reachable),
            transitions=filtered,
            current_state=self.current_state
        )

    def _are_dfas_isomorphic(self, dfa1, dfa2):
        (states1, locs1, t1, _, a1) = dfa1
        (states2, locs2, t2, _, a2) = dfa2
        if len(states1) != len(states2) or len(locs1) != len(locs2) or set(a1) != set(a2):
            return False
        
        # For a square with 4 locations, we need to check all symmetries
        if len(locs1) == 4:
            # Define all symmetries of the square
            # Assuming locations are ordered as: [top, right, bottom, left]
            symmetries = [
                # Identity
                lambda x: x,
                # 90° rotation
                lambda x: [x[3], x[0], x[1], x[2]],
                # 180° rotation
                lambda x: [x[2], x[3], x[0], x[1]],
                # 270° rotation
                lambda x: [x[1], x[2], x[3], x[0]],
                # Horizontal reflection
                lambda x: [x[2], x[1], x[0], x[3]],
                # Vertical reflection
                lambda x: [x[0], x[3], x[2], x[1]],
                # Diagonal reflection (top-left to bottom-right)
                lambda x: [x[0], x[1], x[2], x[3]],
                # Anti-diagonal reflection (top-right to bottom-left)
                lambda x: [x[0], x[3], x[2], x[1]]
            ]
            
            # Try each symmetry
            for sym in symmetries:
                loc_map = {locs1[i]: sym(locs1)[i] for i in range(4)}
                if self._check_transition_equivalence(states1, t1, t2, loc_map):
                    return True
        else:
            # For non-square gadgets, just check identity and reverse
            n = len(locs1)
            # Identity mapping
            loc_map = {l: l for l in locs1}
            if self._check_transition_equivalence(states1, t1, t2, loc_map):
                return True
                
            # Reflection mapping (reverse the locations)
            loc_map = {locs1[i]: locs1[n-1-i] for i in range(n)}
            if self._check_transition_equivalence(states1, t1, t2, loc_map):
                return True
                
        return False

    def _check_transition_equivalence(self, states, t1, t2, loc_map):
        dict1 = {s: {(u,v):w for (u,v,w) in t1.get(s, [])} for s in states}
        dict2 = {s: {(u,v):w for (u,v,w) in t2.get(s, [])} for s in states}
        for s in states:
            mapped = { (loc_map[u], loc_map[v]): w for (u,v),w in dict1[s].items() }
            if mapped != dict2[s]:
                return False
        return True

class GadgetNetwork(GadgetLike):
    def __init__(self, name="GadgetNetwork"):
        self.name = name
        self.subgadgets: List[Gadget] = []
        self.operations: List[Tuple] = []

    def getLocations(self) -> List[int]:
        """Get all locations from all subgadgets."""
        locations = []
        for gadget in self.subgadgets:
            locations.extend(gadget.getLocations())
        return locations

    def getStates(self) -> List[int]:
        """Get all states from all subgadgets."""
        states = []
        for gadget in self.subgadgets:
            states.extend(gadget.getStates())
        return states

    def getTransitions(self) -> Dict[int, List[Tuple[int, int, int]]]:
        """Get all transitions from all subgadgets."""
        transitions = {}
        for gadget in self.subgadgets:
            transitions.update(gadget.getTransitions())
        return transitions

    def __str__(self):
        return '\n'.join(f"{i}: {g}" for i,g in enumerate(self.subgadgets))

    def __iadd__(self, other: GadgetLike):
        self.subgadgets.append(other)
        return self

    def connect(self, gadget_index, loc1, loc2):
        gadget = self.subgadgets[gadget_index]
        self.operations.append(("CONNECT", gadget, loc1, loc2))

    def combine(self, i1, i2, rotation, splice):
        self.operations.append(("COMBINE", i1, i2, rotation, splice))

    def do_connect(self, gadget: Gadget, loc1, loc2):
        # Group transitions by locations
        if loc1 not in gadget.free_ports or loc2 not in gadget.free_ports:
            raise ValueError("Port already in use.")
        inbound1, outbound1, inbound2, outbound2 = {}, {}, {}, {}
        new_trans = {}
        for state, trans_list in gadget.transitions.items():
            new_trans[state] = []
            for li, lo, ns in trans_list:
                if li not in (loc1, loc2) and lo not in (loc1, loc2):
                    new_trans[state].append((li, lo, ns))
                if li==loc1 and lo!=loc2: outbound1.setdefault(state,[]).append((lo,ns))
                if li==loc2 and lo!=loc1: outbound2.setdefault(state,[]).append((lo,ns))
                if lo==loc1 and li!=loc2: inbound1.setdefault(ns,[]).append((li,state))
                if lo==loc2 and li!=loc1: inbound2.setdefault(ns,[]).append((li,state))
        for state in gadget.states:
            for li, st in inbound1.get(state,[]):
                for lo, ns in outbound2.get(state,[]):
                    new_trans[st].append((li, lo, ns))
            for li, st in inbound2.get(state,[]):
                for lo, ns in outbound1.get(state,[]):
                    new_trans[st].append((li, lo, ns))
        gadget.transitions = new_trans
        gadget.locations = [l for l in gadget.locations if l not in (loc1, loc2)]
        gadget.removePorts(loc1, loc2)

    def do_combine(self, i1, i2, rotation, splice):
        g1 = self.subgadgets[i1]
        g2 = self.subgadgets[i2]
        mod = len(g2.locations)
        rot_locs = [((l+rotation)%mod+splice+1) for l in g2.locations]
        rot_trans = {s:[((li+rotation)%mod+splice+1,(lo+rotation)%mod+splice+1,ns) for li,lo,ns in lst]
                     for s,lst in g2.transitions.items()}
        # Build composite
        new_locs = g1.locations[:splice+1] + rot_locs + [l+mod for l in g1.locations[splice+1:]]
        new_states = [(s1,s2) for s1 in g1.states for s2 in g2.states]
        # remap g1 transitions
        g1t = {s:[( (li+mod if li>splice else li),(lo+mod if lo>splice else lo),ns)
                   for li,lo,ns in lst] for s,lst in g1.transitions.items()}
        new_trans = {}
        for s1,s2 in new_states:
            new_trans[(s1,s2)] = []
            for li,lo,ns in g1t[s1]: new_trans[(s1,s2)].append((li,lo,(ns,s2)))
            for li,lo,ns in rot_trans[s2]: new_trans[(s1,s2)].append((li,lo,(s1,ns)))
        new_g = Gadget(
            name=f"Combined({g1.name}+{g2.name})",
            locations=new_locs,
            states=new_states,
            transitions=new_trans,
            current_state=(g1.current_state,g2.current_state)
        )
        # retire inputs
        for idx in sorted([i1,i2], reverse=True): del self.subgadgets[idx]
        self.subgadgets.append(new_g)
        return new_g

    def simplify(self):
       # assert len(self.subgadgets)==1, "Expected single composite"
       # just returns the last one
        return self.subgadgets[-1]

    def do_delete_location(self, gadget: Gadget, loc: int):
        """Delete a location from a gadget by removing all transitions
        referencing it and dropping the label from the gadget."""
        new_trans = {}
        for state, trans_list in gadget.transitions.items():
            new_trans[state] = [
                (li, lo, ns)
                for li, lo, ns in trans_list
                if li != loc and lo != loc
            ]
        gadget.transitions = new_trans
        if loc in gadget.locations:
            gadget.locations.remove(loc)
        gadget.removePorts(loc)
        return gadget
