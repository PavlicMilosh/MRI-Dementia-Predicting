from neat.InovationType import InnovationType
from neat.NeuronGene import NeuronGene
from neat.NeuronType import NeuronType


class Innovation:
    """
    Class represents one innovation.
    """

    def __init__(self,
                 innovation_id: int                 = 0,
                 innovation_type: InnovationType    = None,
                 neuron_in_id: int                  = 0,
                 neuron_out_id: int                 = 0,
                 neuron_id: int                     = 0,
                 neuron_type: NeuronType            = None):

        self.innovation_id = innovation_id
        self.innovation_type = innovation_type
        self.neuron_in_id = neuron_in_id
        self.neuron_out_id = neuron_out_id
        self.neuron_id = neuron_id
        self.neuron_type = neuron_type


    @classmethod
    def init1(cls,
              neuron: NeuronGene,
              innovation_id: int,
              neuron_id: int) -> 'Innovation':

        ret = cls()
        ret.innovation_id = innovation_id
        ret.neuron_id = neuron_id
        ret.neuron_type = neuron.neuron_type
        ret.innovation_type = InnovationType.NEW_NEURON
        ret.neuron_in = -1
        ret.neuron_out = -1
        return ret


    @classmethod
    def init2(cls,
              neuron_in_id: int,
              neuron_out_id: int,
              innovation_type: InnovationType,
              innovation_id: int) -> 'Innovation':
        ret = cls()
        ret.neuron_in_id = neuron_in_id
        ret.neuron_out_id = neuron_out_id
        ret.innovation_type = innovation_type
        ret.innovation_id = innovation_id
        ret.neuron_id = 0
        ret.neuron_type = None
        return ret


    @classmethod
    def init3(cls,
              neuron_in_id: int,
              neuron_out_id: int,
              innovation_type: InnovationType,
              innovation_id: int,
              neuron_type: NeuronType,
              ) -> 'Innovation':

        ret = cls()
        ret.neuron_in = neuron_in_id
        ret.neuron_out_id = neuron_out_id
        ret.innovation_type = innovation_type
        ret.innovation_id = innovation_id
        ret.neuron_type = neuron_type
        return ret

