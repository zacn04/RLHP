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

    combined = net.combine(ap2t1, ap2t2, rotation=0)


    net.connect(combined, 1, 5)  
    net.connect(combined, 2, 6)  

    res = net.simplify()

    net3 = GadgetNetwork()

    net3 += Crossing2Toggle()
    res3 = net3.simplify()


    print(res == res3) # should be True