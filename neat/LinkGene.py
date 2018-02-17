class LinkGene:
    """
    Class represents a link gene.
    """

    def __init__(self,
                 from_neuron_id: int = 0,
                 to_neuron_id: int = 0,
                 weight: float = 0.0,
                 enabled: bool = True,
                 recurrent: bool = False,
                 innovation_id: int = 0):

        self.from_neuron_id = from_neuron_id
        self.to_neuron_id = to_neuron_id
        self.weight = weight
        self.enabled = enabled
        self.recurrent = recurrent
        self.innovation_id = innovation_id
        if not self.enabled:
            print("asdfasdfasdfasfas")


    @classmethod
    def constructor1(cls,
                     neuron_in_id: int,
                     neuron_out_id: int,
                     enabled: bool,
                     innovation_id: int,
                     weight: float,
                     recurrent: bool = False) -> 'LinkGene':
        ret = cls()
        ret.from_neuron_id = neuron_in_id
        ret.to_neuron_id = neuron_out_id
        ret.enabled = enabled
        ret.innovation_id = innovation_id
        ret.weight = weight
        ret.recurrent = recurrent
        if not ret.enabled:
            print("asdfasdfasdfadfdfffadafsdfasdfasdfasdfasdfafafdffsfsdffaafsdafsdasdfsdff")
        return ret


    @classmethod
    def copy(cls, other: 'LinkGene') -> 'LinkGene':
        ret = cls()
        ret.innovation_id = other.innovation_id
        ret.from_neuron_id = other.from_neuron_id
        ret.to_neuron_id = other.to_neuron_id
        ret.weight = other.weight
        ret.recurrent = other.recurrent
        ret.enabled = other.enabled

        return ret


    @classmethod
    def constructor2(cls) -> 'LinkGene':
        return cls()


    def __lt__(self, other):
        return self.innovation_id < other.innovation_id
