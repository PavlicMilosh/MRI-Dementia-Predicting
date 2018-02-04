class LinkGene:

    def __init__(self, from_neuron: int, to_neuron: int, weight: float,
                 enabled: bool, recurrent: bool, innovation_id: int):
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron
        self.weight = weight
        self.enabled = enabled
        self.recurrent = recurrent
        self.innovation_id = innovation_id


    @classmethod
    def constructor(cls, neuron_in, neuron_out, enabled, innovation_id, weight, recurrent):
        ret = cls()
        ret.from_neuron = neuron_in
        ret.to_neuron = neuron_out
        ret.enabled = enabled
        ret.innovation_id = innovation_id
        ret.weight = weight
        ret.recurrent = recurrent
        return ret


    def __lt__(self, other):
        return self.innovation_id < other.innovation_id
