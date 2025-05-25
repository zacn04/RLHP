import sys, os, pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from oop.gadgets.gadgetdefs import *
from oop.gadgets.gadgetlike import *

def test_invalid_combine_and_connect():
    net = GadgetNetwork()
    net += Parallel2Toggle()

    # 1) COMBINE on non-existent gadget → must raise
    with pytest.raises(IndexError):
        net.do_combine(0, 1, 0, 0)

    # 2) CONNECT re-using a port → must raise
    g = net.subgadgets[0]
    net.do_connect(g, 0, 1)
    net.do_connect(g, 2, 3)           # legal first time

    assert net.simplify() != Crossing2Toggle()

    # 3) Cross-network duplicate connect
    net2 = GadgetNetwork()
    net2 += AntiParallel2Toggle()
    net2 += AntiParallel2Toggle()

    net2.do_combine(0, 1, 3, 0)
    g2 = net2.subgadgets[0]
    net2.do_connect(g2, 2, 4)         # first connect

    with pytest.raises(ValueError):
        net2.do_connect(g2, 2, 4)     # duplicate

    # 4) Resulting gadget is not C2T
    assert net2.simplify() != Crossing2Toggle()
