class LinkGene:

    def __init__(self, from_neuron_id: int, to_neuron_id: int, weight: float,
                 enabled: bool, recurrent: bool, innovation_id: int):
        self.from_neuron_id = from_neuron_id
        self.to_neuron_id = to_neuron_id
        self.weight = weight
        self.enabled = enabled
        self.recurrent = recurrent
        self.innovation_id = innovation_id


    @classmethod
    def constructor(cls, neuron_in_id, neuron_out_id, enabled, innovation_id, weight, recurrent):
        ret = cls()
        ret.from_neuron_id = neuron_in_id
        ret.to_neuron_id = neuron_out_id
        ret.enabled = enabled
        ret.innovation_id = innovation_id
        ret.weight = weight
        ret.recurrent = recurrent
        return ret


    def __lt__(self, other):
        return self.innovation_id < other.innovation_id
