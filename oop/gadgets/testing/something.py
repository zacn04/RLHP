# save this as fix_imports.py in the same directory as your test script
import os
import sys

# Get the absolute path to the RLHP project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))

# Add the project root to Python's path
sys.path.insert(0, project_root)

# Now import your modules
from oop.gadgets.gadgetdefs import AntiParallel2Toggle, Crossing2Toggle
from oop.gadgets.gadgetlike import GadgetNetwork, Gadget
from copy import deepcopy
# Your testing code here
# ... 

print("Imports successful!")
print(f"Project root: {project_root}")
print(f"Modules imported: {AntiParallel2Toggle}, {Crossing2Toggle}")

net = GadgetNetwork()
a1 = AntiParallel2Toggle()
net += a1

net2 = GadgetNetwork()
a2 = AntiParallel2Toggle()
net += a2