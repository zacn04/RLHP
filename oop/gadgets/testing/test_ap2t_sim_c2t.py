import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
from oop.gadgets.gadgetdefs import AntiParallel2Toggle, Crossing2Toggle
from oop.gadgets.gadgetlike import GadgetNetwork


def test_ap2t_sim_c2t():
    net = GadgetNetwork()
    
    # Add the first AP2T
    ap2t1 = AntiParallel2Toggle()
    net += ap2t1
    
    ap2t2 = AntiParallel2Toggle()
    ap2t2.setCurrentState(1)
    net += ap2t2
    
    print("Initial network:")
    print(net)
    
    # Use do_combine instead of combine to get the actual gadget
    net.do_combine(0, 1, rotation=0, splice=1)
   
    
    # Now connect, 
    net.connect(-1, 1, 5)
    net.connect(-1, 2, 6)
    
    
    res = net.simplify()
    
    net2 = GadgetNetwork()
    net2 += Crossing2Toggle()
    res2 = net2.simplify()
    print(res, res2)
    
    assert(res == res2)

test_ap2t_sim_c2t()