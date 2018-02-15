from neat.NeuronType import NeuronType


class NeuronGene:
    """
    Class represents a neuron gene.
    """

    def __init__(self,
                 neuron_id: int = 0,
                 neuron_type: NeuronType = None,
                 recurrent: bool = False,
                 activation_response: float = 0,
                 split_x: int = 0,
                 split_y: int = 0):

        self.neuron_id = neuron_id
        self.neuron_type = neuron_type
        self.recurrent = recurrent
        self.activation_response = activation_response
        self.split_x = split_x
        self.split_y = split_y


    @classmethod
    def constructor(cls,
                    neuron_type: NeuronType,
                    neuron_id: int,
                    split_x: float,
                    split_y: float,
                    recurrent: bool,
                    activation: float = 1):
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
