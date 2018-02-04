from neat.util import InnovationType, NeuronType


class Innovation:

    def __init__(self, innovation_id: int, innovation_type: InnovationType, neuron_in: int, neuron_out: int, \
                 neuron_id: int, neuron_type: NeuronType):
        self.innovation_id = innovation_id
        self.innovation_type = innovation_type
        self.neuron_in = neuron_in
        self.neuron_out = neuron_out
        self.neuron_id = neuron_id
        self.neuron_type = neuron_type


