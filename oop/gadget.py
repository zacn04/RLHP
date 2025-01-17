'''
Author: Zac Nwogwugwu, 2024

This will most likely be defining our toolkit for gadgets.
'''

from gadgetlike import GadgetLike

class Gadget(GadgetLike):
    def __init__(self, name=None, locations=None, states=None, transitions=None, current_state=0):
        self.locations = locations
        self.states = states
        self.name = name
        self.transitions = transitions
        self.current_state = current_state

    def __repr__(self):
        return f"{self.name} Gadget is in state {self.current_state}"
    
    def traverse(self, in_location, out_location):
        #Generalised traversal logic
        try:
            self.current_state = self.transitions[self.current_state][(in_location, out_location)]
            return True
        except KeyError:
            return False
            '''raise TraversalError(f"Invalid transition from {in_location} to {out_location} in state {self.current_state}")
        finally:
            pass'''

    def getCurrentState(self):
        return self.current_state
    
    def setCurrentState(self, state):
        states = self.getStates()
        if state in states:
            self.current_state = state
        else:
            raise ValueError("State not found")
    
    def getLocations(self):
        return self.locations
    
    def getStates(self):
        return self.states
    
    def getTransitions(self):
        return self.transitions
    
    def __getitem__(self, transition): #allows for easier transitions.
        self.traverse(transition[0], transition[1])
        
    

class TraversalError(Exception):
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return self.message



class Toggle2Locking(Gadget):
    def __init__(self):
        locations = [0,1,2,3]
        states = [0,1,2]
        name = "Locking 2 Toggle"
        transitions = {
            0: {(0, 1): 1, (2, 3): 2},
            1: {(1, 0): 0},
            2: {(3, 2): 0}
        }
        current_state = 0
        super().__init__(name, locations, states, transitions, current_state)
    

class Door(Gadget):
    def __init__(self):
        locations = [0,1,2,3]
        states = [0,1]
        name = "Door"
        transitions = {
            0: {(0,1): 1},
            1: {(3,2): 0, (1,0): 1} #traversing an open door stays open, can still close it.
        }
        current_state = 0
        super().__init__(name, locations, states, transitions, current_state)

class SelfClosingDoor(Gadget): #basically same as door but any traversal of the open door closes it.
    def __init__(self):
        locations = [0,1,2,3]
        states = [0,1]
        name = "Self Closing Door"
        transitions = {
            0: {(0,1): 1},
            1: {(3,2): 0, (1,0): 0} #traversing an open door closes it, in any state
        }
        current_state = 0
        super().__init__(name, locations, states, transitions, current_state)

class Toggle2(Gadget):
    def __init__(self):
        locations = [0,1,2,3]
        states = [0,1,2]
        name = "2 Toggle"
        transitions = {
            0: {(0, 1): 1, (2, 3): 2},
            1: {(1, 0): 0},
            2: {(3, 2): 0}
        }
        current_state = 0
        super().__init__(name, locations, states, transitions, current_state)

class TripwireLock(Gadget):
    #TODO: implement
    pass

class Diode(Gadget):
    #TODO: implement
    pass

#Tests!
    

X = Toggle2Locking()

print(X)

X[(0,1)]

X[(0,3)]

'''print(X.traverse(1, 2) == True) #should return False

print(X.traverse(1, 4) == True) #should return True'''


print(X.getCurrentState()) #should be 1


X[(0,1)]
X[(3,2)]
X[(3,1)]
X[(3,0)]
'''print(X.traverse(1, 2) == True) #should return False

print(X.traverse(4, 3) == True) #should return False

print(X.traverse(4, 2) == True) #should return False

print(X.traverse(4, 1) == True) #should return True'''

print(X.getCurrentState())

print(X)

X[(1,0)]

print(X)

X[(2,3)]

print(X)

X[(3,2)]

print(X)


#Want to implement logic s.t. 
# if I am in state 0 and I do gadget[(0,1)] (if (0,1) is a transition in state 0, it goes to state 1)
# if it isn't should throw an error.