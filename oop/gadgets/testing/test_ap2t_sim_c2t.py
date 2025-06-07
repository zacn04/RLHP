import sys
import os
from pathlib import Path

# Get the project root directory (RLHP)
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from oop.gadgets.gadgetdefs import AntiParallel2Toggle, Crossing2Toggle
from oop.gadgets.gadgetlike import GadgetNetwork

def test_ap2t_sim_c2t():
    # Create a fresh network
    net = GadgetNetwork()
    
    # Add the first AP2T in state 0
    ap2t1 = AntiParallel2Toggle()
    net += ap2t1
    
    # Add the second AP2T in state 1
    ap2t2 = AntiParallel2Toggle()
    ap2t2.setCurrentState(1)
    net += ap2t2
    
    net.do_combine(0, 1, rotation=0, splice=1)

    # Now connect.
    gadget = net.subgadgets[-1]
    net.do_connect(gadget, 1, 5)
    net.do_connect(gadget, 6, 2)

    res = net.simplify()

    net2 = GadgetNetwork()
    net2 += Crossing2Toggle()
    res2 = net2.simplify()
    print(res, res2)
    
    assert res == res2
if __name__ == "__main__":
    test_ap2t_sim_c2t()
