import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
from oop.gadgets.gadgetdefs import Toggle2, Crossing2Toggle
from oop.gadgets.gadgetlike import GadgetNetwork


def test_c2t_sim_p2t():
    net = GadgetNetwork()
    
    # Add the first C2T
    c2t1 = Crossing2Toggle()
    net += c2t1
    
    c2t2 = Crossing2Toggle()
    net += c2t2
    
    
    net.do_combine(0, 1, rotation=0, splice=2)
   
    
    # Now connect.
    gadget = net.subgadgets[-1]
    net.do_connect(gadget, 2, 4)
    net.do_connect(gadget, 7, 3)
    
    
    res = net.simplify()
    
    net2 = GadgetNetwork()
    net2 += Toggle2()
    res2 = net2.simplify()
    
    assert(res == res2)

test_c2t_sim_p2t()