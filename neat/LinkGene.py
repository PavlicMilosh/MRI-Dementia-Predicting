class LinkGene:

    def __init__(self, from_neuron: int, to_neuron: int, weight: float,
                 enabled: bool, recurrent: bool, innovation_id: int):
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron
        self.weight = weight
        self.enabled = enabled
        self.recurrent = recurrent
        self.innovation_id = innovation_id

    def __lt__(self, other):
        return self.innovation_id < other.innovation_id
