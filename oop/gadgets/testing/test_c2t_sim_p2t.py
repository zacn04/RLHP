from oop.gadgets.gadgetdefs import Crossing2Toggle, Toggle2
from oop.gadgets.gadgetlike import GadgetNetwork


def test_c2t_sim_p2t():
    net = GadgetNetwork()

    # Add the first AP2T
    ap2t1 = Crossing2Toggle()
    net += ap2t1

    ap2t2 = Crossing2Toggle()
    ap2t2.setCurrentState(1)
    net += ap2t2

    combined = net.combine(0, 1, rotation=0, splicing_index=1)
    net.connect(combined, 6, 5)
    net.connect(combined, 1, 2)


    res = net.simplify()


    net2 = GadgetNetwork()

    net2 += Toggle2()


    res2 = net2.simplify()



    assert(res == res2) # should be True