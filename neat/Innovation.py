from neat.NeuronGene import NeuronGene
from neat.util import NeuronType, InnovationType


class Innovation:

    __slots__ = ['neuron_in', 'neuron_out', 'neuron_id', 'neuron_type', 'innovation_id', 'innovation_type']

    def __init__(self, *args):
        if len(args) == 3:
            self.init1(args[0], args[1], args[2])
        elif len(args) == 4:
            self.init2(args[0], args[1], args[2], args[3])
        elif len(args) == 5:
            self.init3(args[0], args[1], args[2], args[3], args[4])

    @classmethod
    def init1(cls, neuron: NeuronGene, innovation_id: int, neuron_id: int):
        ret = cls()
        ret.innovation_id = innovation_id
        ret.neuron_id = neuron_id
        ret.neuron_type = neuron.neuron_type
        ret.neuron_in = -1
        ret.neuron_out = -1

    @classmethod
    def init2(cls, n_in: int, n_out: int, innovation_type: InnovationType, innovation_id: int):
        ret = cls()
        ret.neuron_in = n_in
        ret.neuron_out = n_out
        ret.innovation_type = innovation_type
        ret.innovation_id = innovation_id
        ret.neuron_id = 0
        ret.neuron_type = None

    @classmethod
    def init3(cls, n_in: int, n_out: int, innovation_type: InnovationType, innovation_id: int, neuron_type: NeuronType):
        ret = cls()
        ret.neuron_in = n_in
        ret.neuron_out = n_out
        ret.innovation_type = innovation_type
        ret.innovation_id = innovation_id
        ret.neuron_type = neuron_type


class InnovationDB:

    def __init__(self, innovations, next_neuron_id, next_innovation_num):
        self.innovations = innovations
        self.next_neuron_id = next_neuron_id
        self.next_innovation_num = next_innovation_num

    def check_innovation(self, n_in, n_out, innovation_type):
        for innovation in self.innovations:
            if innovation.neuron_in == n_in and innovation.neuron_out == n_out and innovation.innovation_type == innovation_type:
                return innovation.innovation_id
        return -1

    def create_new_innovation(self, n_in, n_out, innovation_type):
        innovation = Innovation(n_in, n_out, innovation_type, self.next_innovation_num)
        if innovation_type == InnovationType.NEW_NEURON:
            innovation.neuron_id = self.next_neuron_id
            self.next_neuron_id += 1

        self.innovations.append(innovation)
        self.next_innovation_num += 1
        return self.next_neuron_id - 1

    def create_new_innovation(self, n_from: int, n_to: int, innovation_type: InnovationType, neuron_type: NeuronType, x: float, y: float):
        innovation = Innovation(n_from, n_to, innovation_type, self.next_innovation_num, neuron_type)
        if innovation_type == InnovationType.NEW_NEURON:
            innovation.neuron_id = self.next_neuron_id
            self.next_neuron_id += 1

        self.innovations.append(innovation)
        self.next_innovation_num += 1
        return self.next_neuron_id - 1

    def create_neuron_from_id(self, neuron_id):
        tmp = NeuronGene()
        for innovation in self.innovations:
            if innovation.neuron_id == neuron_id:
                tmp.neuron_type = innovation.neuron_type
                tmp.neuron_id = neuron_id

                return tmp
        return tmp

    def next_number(self, num = 0):
        self.next_innovation_num += num
        return self.next_innovation_num
