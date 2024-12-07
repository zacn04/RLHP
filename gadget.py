class Gadget:
    def __init__(self, name=None, locations=None, states=None):
        self.locations = locations
        self.states = states
        self.name = name
        self.current_state = None

    def __repr__(self):
        return f"{self.name} is in state {self.current_state}"
    
    