"""
Author: Zac Nwogwugwu, 2025
"""
from gadgetlike import Gadget, GadgetNetwork
def do_simulate(gadgets: list[Gadget], combinations, target: Gadget) -> bool:
    """
    Does a combination of gadgets (through connecting or combining) simulate the target?
    Returns TRUE if so.
    """
    return True



# Need some way of defining these combinations...

# This forms the basis of the reinforcement learning step.

# can a combination just be initialising another gadget with a superimposition?

### TESTS ###

from gadgets import *

net = GadgetNetwork()

net += AntiParallel2Toggle()

net += AntiParallel2Toggle()

net.combine(net.subgadgets[0], net.subgadgets[1], rotation=0)

res = net.canonicalise()
print(res)
print(res.getLocations())
print(res.getTransitions())


# Combined APT!




