import gymnasium as gym
from gymnasium import Env, spaces
import numpy as np
from copy import deepcopy
from oop.gadgets.gadgetlike import GadgetNetwork
from oop.gadgets.gadgetlike import Gadget as g

import logging
import difflib


class GadgetSimulationEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, initial_gadgets, target_gadget, max_steps=8):
        super().__init__()

        self.initial_gadgets = initial_gadgets
        self.target_gadget = target_gadget
        self.max_steps = max_steps

        self.network = None
        self.current_step = 0

        self.max_gadgets = 2
        self.base_ports = max(len(g.getLocations()) for g in self.initial_gadgets)
        self.max_ports = self.max_gadgets * self.base_ports
        self.max_states = 4

        self.num_setstate_ops = self.max_gadgets * self.max_states
        self.num_combine_ops = self.max_gadgets * self.max_gadgets * 4 * self.base_ports #each gadget pair, rotation and splicing indices
        self.num_connect_ops = self.max_gadgets * self.base_ports * self.base_ports # each gadget, each location #
        self.action_space = spaces.Discrete(self.num_combine_ops + self.num_connect_ops + self.num_setstate_ops + 1) #add 1 for STOP!

        self.observation_space = spaces.Dict({
            'state_vector': spaces.Box(low=0, high=1, shape=(32,), dtype=np.float32),
            'action_mask': spaces.MultiBinary(self.action_space.n)
        })

        self.logger = logging.getLogger(self.__class__.__name__)          

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.network = GadgetNetwork()
        for g in deepcopy(self.initial_gadgets):
            self.network += g
        self.current_step = 0
        return self._get_obs(), {}
    

    def step(self, action):
        done = False
        truncated = False
        reward = 0
        info = {}

        if action == self.action_space.n - 1:
            done = True
            simp = deepcopy(self.network).simplify()
            reward = 200 if simp == self.target_gadget else -20
            return self._get_obs(), reward, done, truncated, info
        else:
            try:
                # Check if action is valid before trying it
                if action < self.num_combine_ops:
                    # Check combine validity
                    flat = action
                    ij = flat // (4 * self.base_ports)
                    rem = flat % (4 * self.base_ports)
                    rot = rem // self.base_ports
                    splice = rem % self.base_ports
                    i = ij // self.max_gadgets
                    j = ij % self.max_gadgets
                    
                    if i >= len(self.network.subgadgets) or j >= len(self.network.subgadgets):
                        raise ValueError("Invalid COMBINE: index out of range")
                    if i == j:
                        raise ValueError("Invalid COMBINE: same index")
                    if splice >= len(self.network.subgadgets[i].locations):
                        raise ValueError(f"Invalid COMBINE: splice index {splice} out of range")
                        
                elif action < self.num_combine_ops + self.num_connect_ops:
                    # Check connect validity
                    conn_idx = action - self.num_combine_ops
                    gadget_idx = conn_idx // (self.base_ports * self.base_ports)
                    rem = conn_idx % (self.base_ports * self.base_ports)
                    loc1 = rem // self.base_ports
                    loc2 = rem % self.base_ports
                    
                    if gadget_idx >= len(self.network.subgadgets):
                        raise ValueError("Invalid CONNECT: gadget index out of range")
                    
                    g = self.network.subgadgets[gadget_idx]
                    ports = g.getLocations()
                    
                    if loc1 >= len(ports) or loc2 >= len(ports):
                        raise ValueError(f"Invalid CONNECT: port indices {loc1},{loc2} out of range for ports {ports}")
                    if loc1 == loc2:
                        raise ValueError("Invalid CONNECT: cannot connect port to itself")
                        
                elif action < self.num_combine_ops + self.num_connect_ops + self.num_setstate_ops:
                    # Check set state validity
                    set_idx = action - self.num_combine_ops - self.num_connect_ops
                    g_idx = set_idx // self.max_states
                    s = set_idx % self.max_states
                    
                    if g_idx >= len(self.network.subgadgets):
                        raise ValueError("Invalid SET_STATE: gadget index out of range")
                    
                    g = self.network.subgadgets[g_idx]
                    if s not in g.getStates():
                        raise ValueError(f"Invalid SET_STATE: state {s} not in {g.getStates()}")
                    if s == g.getCurrentState():
                        raise ValueError(f"Invalid SET_STATE: already in state {s}")
                
                # If we get here, the action is valid
                self._apply_action(action)
                reward -=1 # stop doing stuff bro
            except Exception as e:
                info['error'] = str(e)
                reward = -50  # Small penalty for invalid action
                done = False # does not stop, just gets a penalty. 
                
        current = deepcopy(self.network).simplify()
        if current == self.target_gadget: #actually stop ahhh
            return self._get_obs(), 200, True, False, info 
        similarity = self.dfa_similarity(current, self.target_gadget)
        reward += 10 * similarity  # Reward for getting closer to target
        
        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True
            reward -= 1  # Small penalty for running out of steps
        

        return self._get_obs(), reward, done, truncated, info
    
    def dfa_similarity(self, g1, g2):
        """
        Return a [0,1] similarity score between the string forms of g1 and g2,
        using the Ratcliff‐Obershelp “gestalt” algorithm from difflib.SequenceMatcher.
        """
        s1, s2 = str(g1), str(g2)
        # SequenceMatcher.ratio() returns 2*M / T, where M is number of matches
        # and T is total length of both strings.
        return difflib.SequenceMatcher(None, s1, s2).ratio()




    
    def _apply_action(self, action):
        if action < self.num_combine_ops:
            flat = action
            ij = flat // (4 * self.base_ports)
            rem = flat % (4 * self.base_ports)
            rot = rem // self.base_ports
            splice = rem % self.base_ports
            i = ij // self.max_gadgets
            j = ij % self.max_gadgets

            if i >= len(self.network.subgadgets) or j >= len(self.network.subgadgets):
                raise ValueError("Invalid COMBINE: index out of range")
            if i == j:
                raise ValueError("Invalid COMBINE: same index")
            
            self.network.do_combine(i, j, rotation=rot, splice=splice)

        elif action < self.num_combine_ops + self.num_connect_ops:  # CONNECT
            conn_idx = action - self.num_combine_ops
            gadget_idx = conn_idx // (self.base_ports * self.base_ports)
            rem = conn_idx % (self.base_ports * self.base_ports)
            loc1 = rem // self.base_ports       # local index 0..max_ports-1
            loc2 = rem % self.base_ports

            g = self.network.subgadgets[gadget_idx]
            # Map local indices to global port labels:
            ports = g.getLocations()           # e.g. [0,1,2,3,4,5,6,7]
            try:
                port1 = ports[loc1]
                port2 = ports[loc2]
            except IndexError:
                raise ValueError(f"CONNECT: invalid local port {loc1},{loc2} for {ports}")
            if port1 == port2:
                raise ValueError(f"CONNECT: cannot connect port to itself.")
            self.network.do_connect(g, port1, port2)
        elif action <  self.num_combine_ops + self.num_connect_ops + self.num_setstate_ops:
            set_idx = action - self.num_combine_ops - self.num_connect_ops
            g_idx = set_idx // self.max_states
            s = set_idx % self.max_states
            if g_idx >= len(self.network.subgadgets):
                raise ValueError("Invalid gadget index in SET_STATE")
            g = self.network.subgadgets[g_idx]
            if s not in g.getStates():
                raise ValueError(f"Invalid state {s} for gadget {g}")
            if s == g.getCurrentState():
                raise ValueError(f"Cannot change to same state {s}")
            g.setCurrentState(s)
        else:
            # STOP
            pass


    def action_from_op(self, op):
        if op[0] == "COMBINE":
            _, i, j, rot, splice = op
            idx = (
                (i * self.max_gadgets + j) * (4 * self.base_ports)
                + rot * self.base_ports
                + splice
            )
            return idx

        elif op[0] == "CONNECT":
            _, g_idx, glob1, glob2 = op
            # pick gadget
            if g_idx >= len(self.network.subgadgets):
                raise ValueError(f"Invalid gadget index in op: {op}")
            g = self.network.subgadgets[g_idx]
            # map global ports to local indices 0..3
            locs = g.getLocations()
            try:
                loc1 = locs.index(glob1)
                loc2 = locs.index(glob2)
            except ValueError:
                raise ValueError(f"Port {glob1} or {glob2} not found in {locs}")
            # flatten
            idx = self.num_combine_ops
            for gg in range(self.max_gadgets):
                for l1 in range(self.max_ports):
                    for l2 in range(self.max_ports):
                        if gg == g_idx and l1 == loc1 and l2 == loc2:
                            return idx
                        idx += 1
            raise ValueError(f"Invalid CONNECT op after localizing: {op}")
        elif op[0] == "SET_STATE":
            _, g_idx, s = op
            return self.num_combine_ops + self.num_connect_ops + g_idx * self.max_states + s
        elif op[0] == "STOP":
            return self.action_space.n - 1

        else:
            raise ValueError(f"Unknown op: {op}")

        

    def _build_action_mask(self):
        """
        Returns a (self.action_space.n,) int8 mask where
        mask[k] == 1 if action k is currently valid, else 0.
        """
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        idx = 0

        # 1) COMBINE actions
        # flatten order: i, j, rot, splice
        for i in range(self.max_gadgets):
            for j in range(self.max_gadgets):
                for rot in range(4):
                    for sp in range(self.base_ports):
                        # valid if both gadgets exist, are distinct, and splice < #ports
                        valid = (
                            i < len(self.network.subgadgets)
                            and j < len(self.network.subgadgets)
                            and i != j
                            and sp < len(self.network.subgadgets[i].locations)
                        )
                        mask[idx] = 1 if valid else 0
                        idx += 1

        # 2) CONNECT actions
        # flatten order: gadget g, loc1, loc2
        for g_idx in range(self.max_gadgets):
            ports = self.network.subgadgets[g_idx].getLocations() if g_idx < len(self.network.subgadgets) else []
            for loc1 in range(self.base_ports):
                for loc2 in range(self.base_ports):
                    valid = (loc1 in range(len(ports)) 
                            and loc2 in range(len(ports))
                            and loc1 != loc2)
                    mask[idx] = 1 if valid else 0
                    idx += 1


        for g_idx in range(self.max_gadgets):
            if g_idx < len(self.network.subgadgets):
                states = self.network.subgadgets[g_idx].getStates()
                current = self.network.subgadgets[g_idx].getCurrentState()
            else:
                states, current = [], None
            for s in range(self.max_states):
                valid = (s in states) and (s != current)
                mask[idx] = 1 if valid else 0
                idx += 1

        # 3) STOP action (last index)
        # idx should now equal num_combine_ops + num_connect_ops
        # STOP only valid when the reachable gadget matches target
        simp = deepcopy(self.network).simplify()
        mask[-1] = 1 if simp == self.target_gadget else 0
        if self.network.simplify() == self.target_gadget:
             mask[:-1] = 0
             mask[-1] = 1

        return mask



    def _get_obs(self):
        # 1) allocate
        state_vec = np.zeros(32, dtype=np.float32)
        idx = 0

        # A) one‐hot current state for each of the max_gadgets
        for i in range(self.max_gadgets):
            if i < len(self.network.subgadgets):
                s = self.network.subgadgets[i].getCurrentState()
            else:
                s = None
            for st in range(self.max_states):
                state_vec[idx] = 1.0 if s == st else 0.0
                idx += 1

        # B) port‐presence mask for each gadget
        for i in range(self.max_gadgets):
            if i < len(self.network.subgadgets):
                locs = self.network.subgadgets[i].getLocations()
            else:
                locs = []
            for p in range(self.base_ports):
                state_vec[idx] = 1.0 if p in locs else 0.0
                idx += 1

        # C) fraction of steps used
        state_vec[idx] = self.current_step / float(self.max_steps)
        idx += 1

        # D) similarity scalar
        sim = self.dfa_similarity(self.network.simplify(), self.target_gadget)
        state_vec[idx] = sim
        idx += 1

        # (idx should now be ≤ 32; the rest stays zero)
        mask = self._build_action_mask()
        return {"state_vector": state_vec, "action_mask": mask}

