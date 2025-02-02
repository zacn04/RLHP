"""
Author: Zac Nwogwugwu, 2025
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from oop.gadgets.gadgetlike import GadgetNetwork
from oop.gadgets.gadgetdefs import *
from oop.dfa.hopcroft import hopcroft_minimisation

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


print(res == res2) # should be True
# # print(res == res3) # should be True