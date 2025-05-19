import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from oop.gadgets.gadgetdefs import *
from oop.gadgets.gadgetlike import *

def test_failure():

    gadget1 = Toggle2()

    gadget2 = Toggle2()

    network = GadgetNetwork()

    network += gadget1
    try:
        network.do_combine(0, 1, 0, 0) # should fail
        assert False, "Expected IndexError due to invalid combine indices"
    except IndexError:
        print("caught expected IndexError.")
    gadget = network.subgadgets[0]
    network.do_connect(gadget, 0, 1)
    network.do_connect(gadget, 2, 3) # should have no transitions

    res = network.simplify()
    assert isinstance(res, Gadget)
    assert res != gadget2


if __name__ == "__main__":
    test_failure()