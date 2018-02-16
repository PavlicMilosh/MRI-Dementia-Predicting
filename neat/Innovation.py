from neat.InnovationType import InnovationType
from neat.NeuronGene import NeuronGene
from neat.NeuronType import NeuronType


class Innovation:
    """
    Class represents one innovation.
    """

    def __init__(self,
                 innovation_id: int = 0,
                 innovation_type: InnovationType = None,
                 neuron_from_id: int = 0,
                 neuron_to_id: int = 0,
                 neuron_id: int = 0,
                 neuron_type: NeuronType = None):

        self.innovation_id = innovation_id
        self.innovation_type = innovation_type
        self.neuron_from_id = neuron_from_id
        self.neuron_to_id = neuron_to_id
        self.neuron_id = neuron_id
        self.neuron_type = neuron_type


    @classmethod
    def create_neuron_innovation(cls,
                                 neuron_id: int,
                                 from_neuron_id: int,
                                 to_neuron_id: int,
                                 innovation_id: int,
                                 neuron_type: NeuronType
                                 ) -> 'Innovation':
        ret = cls()
        ret.neuron_id = neuron_id
        ret.neuron_from_id = from_neuron_id
        ret.neuron_to_id = to_neuron_id
        ret.innovation_type = InnovationType.NEW_NEURON
        ret.innovation_id = innovation_id
        ret.neuron_type = neuron_type
        return ret


    @classmethod
    def create_link_innovation(cls,
                               neuron_from_id: int,
                               neuron_to_id: int,
                               innovation_id: int
                               ) -> 'Innovation':
        ret = cls()
        ret.neuron_from_id = neuron_from_id
        ret.neuron_to_id = neuron_to_id
        ret.innovation_type = InnovationType.NEW_LINK
        ret.innovation_id = innovation_id
        ret.neuron_id = -1
        ret.neuron_type = None
        return ret
