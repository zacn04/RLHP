from oop.gadgets.gadgetdefs import Crossing2Toggle, Toggle2
from oop.gadgets.gadgetlike import GadgetNetwork


def test_c2t_sim_p2t():
    net = GadgetNetwork()
    
    # Add the first C2T
    ap2t1 = Crossing2Toggle()
    net += ap2t1
    
    ap2t2 = Crossing2Toggle()
    net += ap2t2
    
    # Use the combine operation and ADD the result to the network
    combined = net.do_combine(0, 1, rotation=0, splicing_index=0)
    net += combined  # Add the combined gadget to the network
    
    # Now connect using the index of the combined gadget (which is 2)
    net.connect(2, 6, 1)
    net.connect(2, 5, 4)
    
    res = net.simplify()
    
    net2 = GadgetNetwork()
    net2 += Toggle2()
    res2 = net2.simplify()
    
    assert(res == res2)  # should be True