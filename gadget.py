class Gadget:
    def __init__(self, name=None, locations=None, states=None):
        self.locations = locations
        self.states = states
        self.name = name
        self.current_state = None

    def __repr__(self):
        return f"{self.name} is in state {self.current_state}"
    
    def traverse(self, in_location, out_location):
        raise NotImplementedError("Gadgets must be able to be traversed!")


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
    

class TraversalError(Exception):
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return self.message



class Toggle2Locking(Gadget):
    def __init__(self):
        super.__init__()

    def traverse(self, in_location, out_location):
        #If I go from L1 to R1, I go from state 0 to state 1. (C.f. L2 to R2, state 2). [Locations are usually clockwise?]
        #So L1 = 1, L2 = 2, R2=3, R1=4
        # This stops me from doing R2 (c.f R1), but if this is traversed, we reset.

        #Returns True if possible, False otherwise

        initial_state = self.getCurrentState()

        if initial_state == 0:
            if in_location == 1 and out_location == 4: #define allowed transitions, maybe impl. as a function
                self.setCurrentState(1)
            elif in_location == 2 and out_location == 3:
                self.setCurrentState(2)
            else:
                return False

        elif initial_state == 1:
            if in_location == 4 and out_location == 1:
                self.setCurrentState(0)
            else:
                return False
        elif initial_state == 2:
            if in_location == 3 and out_location == 2:
                self.setCurrentState(0)
            else:
                return False

        return True
    
    