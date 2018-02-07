from neat.NeuronType import NeuronType


class NeuronGene:

    def __init__(self,
                 neuron_id: int,
                 neuron_type: NeuronType,
                 recurrent: bool,
                 activation_response: float,
                 split_x: int,
                 split_y: int):

        self.neuron_id = neuron_id
        self.neuron_type = neuron_type
        self.recurrent = recurrent
        self.activation_response = activation_response
        self.split_x = split_x
        self.split_y = split_y


    @classmethod
    def constructor(cls, neuron_type: NeuronType, neuron_id: int, split_x: float, split_y: float,
                    recurrent: bool = False, activation: float = 1):
        ret = cls()
        ret.neuron_type = neuron_type
        ret.neuron_id = neuron_id
        ret.split_x = split_x
        ret.split_y = split_y
        ret.activation_response = activation
        ret.recurrent = recurrent

    @classmethod
    def constructor(cls) -> 'NeuronGene':
        return cls()
