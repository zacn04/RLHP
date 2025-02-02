from oop.gadgets.gadgetdefs import Crossing2Toggle, Toggle2
from oop.gadgets.gadgetlike import GadgetNetwork


def test_c2t_sim_p2t():
    """Test that C2Ts can simulate a P2T (from Lemma 4.3)"""
    net = GadgetNetwork()

    # Add the first AP2T
    ap2t1 = Crossing2Toggle()
    net += ap2t1

    ap2t2 = Crossing2Toggle()
    ap2t2.setCurrentState(1)
    net += ap2t2

    combined = net.combine(ap2t1, ap2t2, rotation=0, splicing_index=1)
    net.connect(combined, 6, 5)
    net.connect(combined, 1, 2)


    res = net.simplify()


    net2 = GadgetNetwork()
    # net2 += Crossing2Toggle()

    net2 += Toggle2()


    res2 = net2.simplify()




    # # net3 = GadgetNetwork()
    # # # net2 += Crossing2Toggle()

    # # net3 += Toggle2Locking()
    # # res3 = net3.simplify()



    print(res.states)
    print(res.locations)
    print(res.transitions)

    print(res2.states)
    print(res2.locations)
    print(res2.transitions)



    # # print(res3.states)
    # # print(res3.locations)
    # # print(res3.transitions)


    assert(res == res2) # should be True