import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
from oop.gadgets.gadgetdefs import AntiParallel2Toggle, NoncrossingWireToggle
from oop.gadgets.gadgetlike import GadgetNetwork


def test_nwt_sim_ap2t():
    net = GadgetNetwork()
    
    # Add the first NWT
    nwt1 = NoncrossingWireToggle()
    net += nwt1
    
    nwt2 = NoncrossingWireToggle()
    net += nwt2
    
    
    net.do_combine(0, 1, rotation=0, splice=3)
   
    
    # Now connect.
    gadget = net.subgadgets[-1]
    net.do_connect(gadget, 2, 5)
    net.do_connect(gadget, 1, 6)
    
    
    res = net.simplify()
    
    net2 = GadgetNetwork()
    net2 += AntiParallel2Toggle()
    res2 = net2.simplify()
    print(res, res2)
    
    assert(res == res2)

#test_nwt_sim_ap2t()