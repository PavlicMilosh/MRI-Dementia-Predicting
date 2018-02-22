from neat.NeuronType import NeuronType


class NeuronGene:
    """
    Class represents a neuron gene.
    """

    def __init__(self,
                 neuron_id: int = 0,
                 neuron_type: NeuronType = None,
                 recurrent: bool = False,
                 activation_response: float = 0.0,
                 innovation_id: int = -1):

        self.neuron_id = neuron_id
        self.neuron_type = neuron_type
        self.recurrent = recurrent
        self.activation_response = activation_response
        self.innovation_id = innovation_id


    @classmethod
    def copy(cls, other: 'NeuronGene') -> 'NeuronGene':
        ret = cls()
        ret.neuron_id = other.neuron_id
        ret.neuron_type = other.neuron_type
        ret.recurrent = other.recurrent
        ret.activation_response = other.activation_response
        ret.innovation_id = other.innovation_id
        return ret


    @classmethod
    def constructor1(cls,
                     neuron_type: NeuronType,
                     neuron_id: int,
                     innovation_id: int,
                     recurrent: bool = False,
                     activation: float = 1):
        ret = cls()
        ret.neuron_type = neuron_type
        ret.neuron_id = neuron_id
        ret.activation_response = activation
        ret.recurrent = recurrent
        ret.innovation_id = innovation_id
        return ret


    @classmethod
    def constructor2(cls) -> 'NeuronGene':
        return cls()


    @classmethod
    def constructor3(cls, neuron_id) -> 'NeuronGene':
        ret = cls()
        ret.neuron_id = neuron_id
        return ret
