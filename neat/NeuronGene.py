from neat.util import NeuronType


class NeuronGene:

    def __init__(self, neuron_id: int, neuron_type: NeuronType, recurrent: bool, activation_response: float,
                 split_x: int, split_y: int):
        self.neuron_id = neuron_id
        self.neuron_type = neuron_type
        self.recurrent = recurrent
        self.activation_response = activation_response
        self.split_x = split_x
        self.split_y = split_y
