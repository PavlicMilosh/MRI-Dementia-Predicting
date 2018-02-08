from neat.Innovation import Innovation
from neat.NeuronGene import NeuronGene
from neat.LinkGene import LinkGene
from neat.InovationType import InnovationType
from neat.NeuronType import NeuronType
from typing import *


class InnovationDB:

    def __init__(self,
                 innovations,
                 next_neuron_id: int,
                 next_innovation_num: int):

        self.innovations = innovations
        self.next_neuron_id = next_neuron_id
        self.next_innovation_num = next_innovation_num


    @classmethod
    def from_genes(cls, link_genes: Sequence[LinkGene], neuron_genes: Sequence[NeuronGene]) -> 'InnovationDB':

        ret = cls()
        ret.next_neuron_id = 0
        ret.next_innovation_num = 0

        # add neurons
        for neuron_gene in neuron_genes:
            ret.innovations.append(Innovation.init1(neuron_gene, ret.next_innovation_num, ret.next_neuron_id))

        # add links
        for link_gene in link_genes:
            innovation = Innovation.init2(link_gene.from_neuron_id, link_gene.to_neuron_id, InnovationType.NEW_LINK, ret.next_innovation_num)
            ret.innovations.append(innovation)
            ret.next_innovation_num += 1


    def check_innovation(self,
                         neuron_in_id: int,
                         neuron_out_id: int,
                         innovation_type: InnovationType):

        for innovation in self.innovations:
            if innovation.neuron_in_id == neuron_in_id and innovation.neuron_out_id == neuron_out_id \
                    and innovation.innovation_type == innovation_type:
                return innovation.innovation_id
        return -1


    def create_link_innovation(self,
                               neuron_in_id: int,
                               neuron_out_id: int,
                               innovation_type: InnovationType):

        innovation = Innovation.init2(neuron_in_id, neuron_out_id, innovation_type, self.next_innovation_num)
        if innovation_type == InnovationType.NEW_NEURON:
            innovation.neuron_in_id = self.next_neuron_id
            self.next_neuron_id += 1

        self.innovations.append(innovation)
        self.next_innovation_num += 1
        return self.next_neuron_id - 1


    def create_neuron_innovation(self,
                                 neuron_in_id: int,
                                 neuron_out_id: int,
                                 innovation_type: InnovationType,
                                 neuron_type: NeuronType,
                                 x: float,
                                 y: float):

        innovation = Innovation.init3(neuron_in_id, neuron_out_id, innovation_type, self.next_innovation_num, neuron_type, x, y)

        if innovation_type == InnovationType.NEW_NEURON:
            innovation.neuron_id = self.next_neuron_id
            self.next_neuron_id += 1

        self.innovations.append(innovation)
        self.next_innovation_num += 1

        return self.next_neuron_id - 1


    def create_neuron_from_id(self, neuron_id):
        tmp = NeuronGene.constructor(NeuronType.HIDDEN, 0, 0, 0)

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
