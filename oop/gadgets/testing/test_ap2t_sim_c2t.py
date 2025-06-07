import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from oop.gadgets.gadgetdefs import AntiParallel2Toggle, Crossing2Toggle
from oop.gadgets.gadgetlike import GadgetNetwork


def test_ap2t_sim_c2t():
    net = GadgetNetwork()

    ap1 = AntiParallel2Toggle()
    net += ap1

    ap2 = AntiParallel2Toggle()
    ap2.setCurrentState(1)
    net += ap2

    net.do_combine(0, 1, rotation=0, splice=1)

    gadget = net.subgadgets[-1]
    net.do_connect(gadget, 1, 5)
    net.do_connect(gadget, 6, 2)

    res = net.simplify()

    net2 = GadgetNetwork()
    net2 += Crossing2Toggle()
    res2 = net2.simplify()

    assert res == res2
