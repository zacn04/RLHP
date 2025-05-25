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
        self.illegal_actions = 0

        self.max_gadgets = 2
        self.base_ports = max(len(g.getLocations()) for g in self.initial_gadgets)
        self.max_ports = 8  # Maximum possible ports after combining two gadgets
        self.max_states = 4

        self.num_setstate_ops = self.max_gadgets * self.max_states
        self.num_combine_ops = self.max_gadgets * self.max_gadgets * 4 * self.base_ports #each gadget pair, rotation and splicing indices
        self.num_connect_ops = self.max_gadgets * self.max_ports * self.max_ports # each gadget, each location #
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
        self.prev_similarity = 0.0
        return self._get_obs(), {}
    

    def step(self, action):
        done = False
        truncated = False
        reward = 0
        info = {}

        if self.current_step % 1000 == 0 and self.current_step > 0:
            print(f"[LOG] up to step {self.current_step}, illegal_actions={self.illegal_actions}")

        if action == self.action_space.n - 1: # stop
            done = True
            simp = deepcopy(self.network).simplify()
            similarity = self.dfa_similarity(simp, self.target_gadget)
            reward = 200 if simp == self.target_gadget else -25
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
                    reward += 4
                        
                elif action < self.num_combine_ops + self.num_connect_ops:
                    # Check connect validity
                    conn_idx = action - self.num_combine_ops
                    gadget_idx = conn_idx // (self.max_ports * self.max_ports)
                    rem = conn_idx % (self.max_ports * self.max_ports)
                    loc1 = rem // self.max_ports
                    loc2 = rem % self.max_ports
                    
                    if gadget_idx >= len(self.network.subgadgets):
                        raise ValueError("Invalid CONNECT: gadget index out of range")
                    
                    g = self.network.subgadgets[gadget_idx]
                    ports = g.getLocations()

                    port1 = ports[loc1]
                    port2 = ports[loc2]
                    
                    if loc1 >= len(ports) or loc2 >= len(ports):
                        raise ValueError(f"Invalid CONNECT: port indices {loc1},{loc2} out of range for ports {ports}")
                    if loc1 == loc2:
                        raise ValueError("Invalid CONNECT: cannot connect port to itself")
                    if port1 not in g.free_ports or port2 not in g.free_ports:
                        #cutting it off
                        raise ValueError("you just cant do this tbh")
                        
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
                    reward += 5
                # If we get here, the action is valid
                decoded = self.op_from_action(action)
                #print(f"[PRE-APPLY] step={self.current_step}, action={decoded}")
                self._apply_action(action)
                #reward -=1 # stop doing stuff bro
            except Exception as e:
                self.illegal_actions +=1
                info['error'] = str(e)
                print(f"[ILLEGAL] step={self.current_step}, action={self.op_from_action(action)}, error={e}")
                return self._get_obs(), -500, True, False, {
                    **info, 
                    "illegal_actions": self.illegal_actions
                 } # ur done tbh
                
        current = deepcopy(self.network).simplify()
        similarity = self.dfa_similarity(current, self.target_gadget)
        delta = similarity - self.prev_similarity

        if delta > 0:
            reward += 10 * delta  # Reward for getting closer to target
        '''if delta < 0: 
            reward += 5 * delta'''
        self.prev_similarity = similarity

        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True
            reward -= 1  # Small penalty for running out of steps
        

        return self._get_obs(), reward, done, truncated, {
            **info,
            "illegal_actions": self.illegal_actions
        }
    
    def dfa_similarity(self, g1, g2):
        """
        Return a [0,1] similarity score between the string forms of g1 and g2,
        using the Ratcliff‐Obershelp "gestalt" algorithm from difflib.SequenceMatcher.
        """
        def transition_string(g):
            parts = []
            for s, lst in sorted(g.getTransitions().items()):
                for (u,v,w) in sorted(lst):
                    parts.append(f"{s}->{u},{v}->{w}")
            return "|".join(parts)
        # SequenceMatcher.ratio() returns 2*M / T, where M is number of matches
        # and T is total length of both strings.
        return difflib.SequenceMatcher(None, transition_string(g1), transition_string(g2)).ratio()




    
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
            gadget_idx = conn_idx // (self.max_ports * self.max_ports)
            rem = conn_idx % (self.max_ports * self.max_ports)
            loc1 = rem // self.max_ports       # local index 0..max_ports-1
            loc2 = rem % self.max_ports

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
            #print("[DEBUG] about to CONNECT labels", port1, port2, " | free_ports =", g.free_ports)
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
            _, g_idx, port1, port2 = op
            g = self.network.subgadgets[g_idx]
            locs = g.getLocations()

            try:
                loc1 = locs.index(port1)
                loc2 = locs.index(port2)
            except ValueError as e:
                raise ValueError(f"Port(s) {port1} or {port2} not found in {locs}")

            idx = self.num_combine_ops + g_idx * self.max_ports * self.max_ports + loc1 * self.max_ports + loc2
            return idx
        elif op[0] == "SET_STATE":
            _, g_idx, s = op
            return self.num_combine_ops + self.num_connect_ops + g_idx * self.max_states + s
        elif op[0] == "STOP":
            return self.action_space.n - 1

        else:
            raise ValueError(f"Unknown op: {op}")

    def op_from_action(env, action: int):
        """
        Reverse maps an action index back to its original operation tuple:
        ("STOP",) or ("SET_STATE", g_idx, s) or ("CONNECT", g_idx, port1, port2) or ("COMBINE", i, j, rot, splice)
        """
        if action == env.action_space.n - 1:
            return ("STOP",)

        elif action < env.num_combine_ops:
            flat = action
            ij = flat // (4 * env.base_ports)
            rem = flat % (4 * env.base_ports)
            rot = rem // env.base_ports
            splice = rem % env.base_ports
            i = ij // env.max_gadgets
            j = ij % env.max_gadgets
            return ("COMBINE", i, j, rot, splice)

        elif action < env.num_combine_ops + env.num_connect_ops:
            conn_idx = action - env.num_combine_ops
            g_idx = conn_idx // (env.max_ports * env.max_ports)
            rem = conn_idx % (env.max_ports * env.max_ports)
            loc1 = rem // env.max_ports
            loc2 = rem % env.max_ports
            try:
                port1 = env.network.subgadgets[g_idx].getLocations()[loc1]
                port2 = env.network.subgadgets[g_idx].getLocations()[loc2]
            except Exception:
                port1, port2 = loc1, loc2  # fallback if out of range
            return ("CONNECT", g_idx, port1, port2)

        elif action < env.num_combine_ops + env.num_connect_ops + env.num_setstate_ops:
            set_idx = action - env.num_combine_ops - env.num_connect_ops
            g_idx = set_idx // env.max_states
            s = set_idx % env.max_states
            return ("SET_STATE", g_idx, s)

        raise ValueError(f"Invalid action index {action} (max {env.action_space.n})")


    def _build_action_mask(self):
        """
        Returns a (self.action_space.n,) int8 mask where
        mask[k] == 1 if action k is currently valid, else 0.
        Ensures at least one action is valid at all times.
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
        # --- 2)  CONNECT(g, loc1, loc2)  (loc1,loc2 ordered) -------------
        for g_idx in range(self.max_gadgets):
            if g_idx < len(self.network.subgadgets):
                ports = self.network.subgadgets[g_idx].getLocations()
                for i, port1 in enumerate(ports):
                    for j, port2 in enumerate(ports):
                        valid = (
                            port1 != port2
                            and port1 in self.network.subgadgets[g_idx].free_ports
                            and port2 in self.network.subgadgets[g_idx].free_ports
                        )
                        mask[idx] = 1 if valid else 0
                        idx += 1
            else:
                idx += self.max_ports * self.max_ports


        # 3) SET_STATE actions
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

        # 4) STOP action (last index)
        # STOP is valid when the reachable gadget matches target
        simp = deepcopy(self.network).simplify()

        # If we've reached the target, only allow STOP
        if simp == self.target_gadget:
            mask[:-1] = 0
            mask[-1] = 1
        # If no actions are valid, allow STOP as a fallback
        elif not np.any(mask):
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
        curr = self.network.simplify()
        sim = self.dfa_similarity(curr, self.target_gadget)
        state_vec[idx] = sim
        idx += 1

        # (idx should now be ≤ 32; the rest stays zero)
        mask = self._build_action_mask()
        return {"state_vector": state_vec, "action_mask": mask}
    
    def local_to_label(self, g_idx: int, loc_idx: int) -> int:
        """Return the physical port label for the current local index."""
        return self.network.subgadgets[g_idx].getLocations()[loc_idx]

