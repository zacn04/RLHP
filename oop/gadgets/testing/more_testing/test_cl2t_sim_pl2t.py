import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
from oop.gadgets.gadgetdefs import ParallelLocking2Toggle, CrossingLocking2Toggle
from oop.gadgets.gadgetlike import GadgetNetwork


def test_cl2t_sim_pl2t():
    net = GadgetNetwork()
    
    # Add the first C2T
    c2t1 = CrossingLocking2Toggle()
    net += c2t1
    
    c2t2 = CrossingLocking2Toggle()
    net += c2t2
    
    
    net.do_combine(0, 1, rotation=0, splice=3)
   
    
    # Now connect.
    gadget = net.subgadgets[-1]
    net.do_connect(gadget, 2, 7)
    net.do_connect(gadget, 1, 4)
    
    
    res = net.simplify()
    
    net2 = GadgetNetwork()
    net2 += ParallelLocking2Toggle()
    res2 = net2.simplify()
    
    #assert(res == res2)

test_cl2t_sim_pl2t()

# Not currently simulatable: requires intermediate location between gadgets
# Would require shared connection point across subgadgets (future extension)
