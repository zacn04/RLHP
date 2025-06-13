import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from oop.gadgets.gadgetdefs import *

def test_trivial_equivalence():

    gadget1 = Parallel2Toggle()

    gadget2 = Parallel2Toggle()

    # They should be the same regardless of state, we test both.

    assert gadget1 == gadget2

    gadget2.setCurrentState(1)

    assert gadget1 == gadget2 


def test_trivial_simulation():
    
    net = GadgetNetwork()
    net += AntiParallel2Toggle()
    net += AntiParallel2Toggle()
    net.connect(0, 1, 2) # self-connection, does notthing

    res = net.simplify()

    assert res != Parallel2Toggle()
    assert res != ParallelLocking2Toggle()

def test_reflection_of_c2t():

    net = GadgetNetwork()
    rotated_c2t = Gadget(
        name = "Rotated C2T",
        locations=[0,1,2,3],
        states=[0,1],
        transitions={
            0: [(0, 2, 1), (3, 1, 1)],
            1: [(2, 0, 0), (1, 3, 0)]
        },
        current_state = 0
    )
    net += rotated_c2t

    res = net.simplify()

    assert res == Crossing2Toggle()

def test_should_not_be_equivalent():
    net = GadgetNetwork()
    net += Parallel2Toggle()
    
    net2 = GadgetNetwork()
    net2 += AntiParallel2Toggle()

    res1 = net.simplify()
    res2 = net2.simplify()
    assert res1 != res2



if __name__ == "__main__":
    test_trivial_equivalence()
    test_trivial_simulation()
    test_reflection_of_c2t()
    test_should_not_be_equivalent()
    print("All tests passed.")