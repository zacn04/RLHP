import sys
import os
from pathlib import Path

# Get the project root directory (RLHP)
project_root = Path(__file__).resolve().parents[3]  # Go up 3 levels
sys.path.append(str(project_root))
from oop.gadgets.gadgetdefs import *
from oop.gadgets.gadgetlike import GadgetNetwork


def test_apl2t_sim_cl2t():
    net = GadgetNetwork()
    
    # Add first APL2T
    apl2t1 = AntiParallelLocking2Toggle()
    net += apl2t1
    
    apl2t2 = AntiParallelLocking2Toggle()
    net += apl2t2
    
    # Use do_combine and add the result to the network
    combined = net.do_combine(0, 1, rotation=2, splicing_index=3)
    net += combined  # Add the combined gadget to the network
    
    # Connect using the index of the combined gadget (which is 2)
    net.connect(2, 1, 6)
    net.connect(2, 2, 5)
    
    res = net.simplify()
    print(res)
    
    net2 = GadgetNetwork()
    net2 += CrossingLocking2Toggle()
    res2 = net2.simplify()
    
    assert(res == res2)

test_apl2t_sim_cl2t()