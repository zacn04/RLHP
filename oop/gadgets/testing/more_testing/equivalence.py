import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from oop.gadgets.gadgetdefs import *

def test_trivial_equivalence():

    gadget1 = Toggle2()

    gadget2 = Toggle2()

    # They should be the same regardless of state, we test both.

    assert gadget1 == gadget2

    gadget2.setCurrentState(1)

    assert gadget1 == gadget2 

    '''def test_square_gadgets():
        # for each individiual gadget type, we check pairwise equivalence
        gadget_types = [
            ParallelLocking2Toggle(), Door(), SelfClosingDoor(), Toggle2(),
            AntiParallel2Toggle(), Crossing2Toggle(), AntiParallelLocking2Toggle(), CrossingLocking2Toggle()
        ]
        for gadget1 in gadget_types:
            for gadget2 in gadget_types:
                if gadget1.name == gadget2.name:
                    continue
                assert not gadget1 == gadget2, f"{gadget1.name} should not be equal to {gadget2.name}"
    '''

def test_trivial_simulation():
    
    net = GadgetNetwork()
    net += AntiParallel2Toggle()
    net += AntiParallel2Toggle()
    net.connect(0, 1, 2) # self-connection, does notthing

    res = net.simplify()

    assert res != Toggle2()
    assert res != ParallelLocking2Toggle()



if __name__ == "__main__":
    test_trivial_equivalence()
    test_trivial_simulation()