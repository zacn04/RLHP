"""
Author: Zac Nwogwugwu, 2025
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from oop.gadgets.gadgetlike import GadgetNetwork
from oop.gadgets.gadgetdefs import *
from oop.dfa.hopcroft import hopcroft_minimisation

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

print(res.locations)
print(res.transitions)
print(res.states)


net2 = GadgetNetwork()
# net2 += Crossing2Toggle()

net2 += AntiParallel2Toggle()


res2 = net2.simplify()

print(res2.locations)
print(res2.transitions)
print(res2.states)



net3 = GadgetNetwork()
# net2 += Crossing2Toggle()

net3 += Crossing2Toggle()
res3 = net3.simplify()
print(res3.locations)
print(res3.transitions)
print(res3.states)

print(res == res2) # should be False
print(res == res3) # should be True



