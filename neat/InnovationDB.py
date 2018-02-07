from neat.Innovation import Innovation
from neat.NeuronGene import NeuronGene
from neat.util import InnovationType, NeuronType


class InnovationDB:

    def __init__(self, innovations, next_neuron_id, next_innovation_num):
        self.innovations = innovations
        self.next_neuron_id = next_neuron_id
        self.next_innovation_num = next_innovation_num


    def check_innovation(self, n_in: int, n_out: int, innovation_type: InnovationType):
        for innovation in self.innovations:
            if innovation.neuron_in == n_in and innovation.neuron_out == n_out \
                    and innovation.innovation_type == innovation_type:
                return innovation.innovation_id
        return -1


    def create_link_innovation(self, n_in: int, n_out: int, innovation_type: InnovationType):
        innovation = Innovation.init2(n_in, n_out, innovation_type, self.next_innovation_num)
        if innovation_type == InnovationType.NEW_NEURON:
            innovation.neuron_id = self.next_neuron_id
            self.next_neuron_id += 1

        self.innovations.append(innovation)
        self.next_innovation_num += 1
        return self.next_neuron_id - 1

    def create_neuron_innovation(self, n_in: int, n_out: int, innovation_type: InnovationType, neuron_type: NeuronType,
                                 x: float, y: float):
        innovation = Innovation.init3(n_in, n_out, innovation_type, neuron_type, x, y)

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


    def next_number(self, num=0):
        self.next_innovation_num += num
        return self.next_innovation_num


    def get_neuron_id(self, inv):
        return self.innovations[inv].neuron_id
