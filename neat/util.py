from enum import Enum

'''ENUMS'''


class NeuronType(Enum):
    INPUT = 1
    BIAS = 2
    HIDDEN = 3
    OUTPUT = 4
    NONE = 5

class InnovationType(Enum):
    NEW_LINK = 1
    NEW_NEURON = 2


'''CONSTANTS'''
chance_of_looped = 0.05
num_tries_to_find_loop = 50
num_tries_to_add_link = 100
