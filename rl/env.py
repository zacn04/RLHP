import gym
from gym import spaces
import numpy as np
from copy import deepcopy
from oop.gadgets.gadgetlike import GadgetNetwork


class GadgetSimulationEnv(gym.Env):
    def __init__(self, initial_gadgets, target_gadget, max_steps=8):
        super().__init__()


        self.initial_gadgets = initial_gadgets
        self.target_gadget = target_gadget
        self.max_steps = max_steps


        self.network = None
        self.current_step = 0

        self.max_gadgets = 6
        self.num_combine_ops = self.max_gadgets * (self.max_gadgets - 1) * 4 * 4 #each gadget pair, rotation and splicing indices
        self.num_connect_ops = self.max_gadgets * 4 * 4 # each gadget, each location
        self.action_space = spaces.Discrete(self.num_combine_ops + self.num_connect_ops + 1) #add 1 for STOP!

        self.observation_space = spaces.Dict({
            'state_vector': spaces.Box(low=0, high=1, shape=(512,), dtype=np.float32),
            'action_mask': spaces.Box(low=0, high=1, shape=(self.action_space.n,), dtype=np.int8)
        })

    def reset(self):
        self.network = GadgetNetwork()
        for g in deepcopy(self.initial_gadgets):
            self.network += g
        self.current_step = 0
        return self._get_obs()
    

    def step(self, action):
        done = False
        reward = 0
        info = {}

        if action == self.action_space.n - 1:
            done = True
            if self.network.simplify() == self.target_gadget:
                reward = 1 #tweak
        else:
            try:
                self._apply_action(action)
            except Exception as e:
                info['error'] = str(e)
                done = True
        
        self.current_step +=1
        if self.current_step >= self.max_steps:
            done = True
        
        return self._get_obs(), reward, done, info
    
    def _apply_action(self, action):
        if action < self.num_combine_ops:
            i = action // (self.max_gadgets - 1) // 16
            rem = action % ((self.max_gadgets - 1) * 16)
            j = rem // 16
            rem2 = rem % 16
            rot = rem2 // 4
            spl = rem2 % 4
            if i == j:
                raise ValueError("Invalid COMBINE: same index")
            self.network.do_combine(i,j, rotation=rot, splice=spl)

        else:
            conn_idx = action - self.num_combine_ops
            gadget_idx = conn_idx // 16
            rem = conn_idx % 16
            loc1 = rem // 4
            loc2 = rem % 4
            if gadget_idx >= len(self.network.subgadgets):
                raise ValueError("Invalid gadget index")
            g = self.network.subgadgets[gadget_idx]
            self.network.do_connect(g, loc1, loc2)


    def action_from_op(self, op):
        """
        Map an expert operation tuple back to the flat action index.
        op is one of:
          ("COMBINE", i, j, rot, splice)
          ("CONNECT", g, loc1, loc2)
          ("STOP",)
        """
        if op[0] == "COMBINE":
            _, i, j, rot, splice = op
            # same ordering as in _get_obs mask and _apply_action decode:
            idx = 0
            for ii in range(self.max_gadgets):
                for jj in range(self.max_gadgets):
                    for r in range(4):
                        for sp in range(4):
                            if ii == i and jj == j and r == rot and sp == splice:
                                return idx
                            idx += 1
            raise ValueError(f"Invalid COMBINE op: {op}")
        
        elif op[0] == "CONNECT":
            _, g, loc1, loc2 = op
            # skip all combines
            idx = self.num_combine_ops
            for gg in range(self.max_gadgets):
                for l1 in range(4):
                    for l2 in range(4):
                        if gg == g and l1 == loc1 and l2 == loc2:
                            return idx
                        idx += 1
            raise ValueError(f"Invalid CONNECT op: {op}")

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
                    for sp in range(4):
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
            for loc1 in range(4):
                for loc2 in range(4):
                    valid = (
                        g_idx < len(self.network.subgadgets)
                        and loc1 in self.network.subgadgets[g_idx].getLocations()
                        and loc2 in self.network.subgadgets[g_idx].getLocations()
                        and loc1 != loc2
                    )
                    mask[idx] = 1 if valid else 0
                    idx += 1

        # 3) STOP action (last index)
        # idx should now equal num_combine_ops + num_connect_ops
        mask[idx] = 1

        # Sanity checks (optional but recommended):
        # assert idx == self.num_combine_ops + self.num_connect_ops
        # assert mask.shape[0] == self.action_space.n

        return mask



    def _get_obs(self):
        state_vector = np.zeros(32, dtype=np.float32)
        mask = self._build_action_mask()
        return {'state_vector': state_vector, 'action_mask': mask}
