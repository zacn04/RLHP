from oop.gadgets.gadgetdefs import *
from oop.gadgets.gadgetlike import GadgetNetwork


def test_cl2t_sim_pl2t():
    net = GadgetNetwork()
    
    cl2t1 = CrossingLocking2Toggle()
    net += cl2t1
    
    cl2t2 = CrossingLocking2Toggle()
    net += cl2t2
    
    # Use do_combine and add the result to the network
    combined = net.do_combine(0, 1, rotation=0, splicing_index=1, reflect=False)
    net += combined  # Add the combined gadget to the network
    
    # Connect using the index of the combined gadget (which is 2)
    net.connect(2, 6, 5)
    net.connect(2, 1, 2)
    
    res = net.simplify()
    
    net2 = GadgetNetwork()
    net2 += ParallelLocking2Toggle()
    res2 = net2.simplify()
    
    assert(res == res2)

test_cl2t_sim_pl2t()