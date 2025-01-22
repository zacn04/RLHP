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

from gadgets import Door, SelfClosingDoor

net = GadgetNetwork(name="Network")
locations = [0, 1, 2, 3, 4, 5]
states = [0, 1, 2]
transitions = {
    0: {(0, 1): 1, (4, 5): 2},
    1: {(1, 2): 2},
    2: {(3, 4): 0}
}
current_state = 0

gadget = Gadget(name="TestGadget", locations=locations, states=states, transitions=transitions, current_state=current_state)
net += gadget
net.connect(net.subgadgets[0], 1,4)

res = net.canonicalise()
print(res)
print(res.getLocations())
print(res.getTransitions())



