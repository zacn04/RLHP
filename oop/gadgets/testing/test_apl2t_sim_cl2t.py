from oop.gadgets.gadgetdefs import *
from oop.gadgets.gadgetlike import GadgetNetwork


def test_apl2t_sim_cl2t():
    net = GadgetNetwork()
    
    # Add first APL2T
    apl2t1 = AntiParallelLocking2Toggle()
    net += apl2t1
    
    apl2t2 = AntiParallelLocking2Toggle() 
    net += apl2t2
    
    
    combined = net.combine(0, 1, rotation=2, splicing_index=3)
    
    
    net.connect(combined, 1, 6)  
    net.connect(combined, 2, 5)
    
    res = net.simplify()

    print(res)
    
    net2 = GadgetNetwork()
    net2 += CrossingLocking2Toggle()
    res2 = net2.simplify()
    
    assert(res == res2)