"""
Author: Zac Nwogwugwu 2025

Handling the logic for gadget networks.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

class GadgetLike:
    """
    The base class for gadgets/gadget networks.
    """
    @abstractmethod
    def get_locations(self) -> List[int]:
        pass

    @abstractmethod
    def get_states(self) -> List[int]:
        pass

    @abstractmethod
    def get_transitions(self) -> Dict[int, Dict[Tuple[int, int], int]]:
        """
        Returns transitions in a canonical format.
        e.g.
        {
        0 : {(1,2): 1} 
        }

        The above is a transition in state 0 from location 1 to location 2, 
        which changes the gadget state to 1
        """
        pass

class GadgetNetwork(GadgetLike):
    def __init__(self, name="GadgetNetwork"):
        """
        The environment in which gadgets will be manipulated.
        """
        self.name = name
        self.subgadgets = []
        self.connections = []

    def connect(self, gadget1, gadget2, loc1, loc2):
        #TODO: Implement
        pass

    def combine(self, gadget1, gadget2, rotation, splicing_index):
        #TODO: Implement
        pass

    def canonicalise():
        """
        The idea is to in some way describe how all the gadgets have been put together.
        """
        #TODO: Implement
        pass

