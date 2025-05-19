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
    combined = net.do_combine(0, 1, rotation=0, splice=1)
    
    # Connect using the index of the combined gadget (which is 2)
    gadget = net.subgadgets[0]
    net.do_connect(gadget, 1, 5)
    net.do_connect(gadget, 2, 6)
    
    res = net.simplify()
    print(res)
    
    net2 = GadgetNetwork()
    net2 += CrossingLocking2Toggle()
    res2 = net2.simplify()
    
    assert(res == res2)

test_apl2t_sim_cl2t()