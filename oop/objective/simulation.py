"""
Author: Zac Nwogwugwu, 2025
"""
from oop.gadgets.gadgetlike import Gadget, GadgetNetwork
from oop.gadgets.gadgetdefs import Toggle2

def do_simulate(gadgets: list[Gadget], combinations, target: Gadget) -> bool:
    """
    Does a combination of gadgets (through connecting or combining) simulate the target?
    Returns TRUE if so.
    """
    network = GadgetNetwork()

    for gadget in gadgets:
        network += gadget
    network.operations = combinations

    proposed = network.canonicalise()
    return are_dfa_equal(proposed, target)

def are_dfa_equal(network1, network2):
    """
    Check if they have an equal number of states, locations.
    Run all states/transitions combinatorially of network2 on one instance of network1
    If they have the same language reutrn True.
    """
    return network1==network2


# Need some way of defining these combinations...

# This forms the basis of the reinforcement learning step.

# can a combination just be initialising another gadget with a superimposition?

### TESTS ###
net = GadgetNetwork()

net += Toggle2()

net += Toggle2()

net.combine(net.subgadgets[0], net.subgadgets[1], rotation=2)

res = net.simplify()

net2 = GadgetNetwork()

net2 += Toggle2()

net2 += Toggle2()

net.combine(net.subgadgets[1], net.subgadgets[0], rotation=0)

res2 = net.simplify()

print(are_dfa_equal(res, res2))
