from neat.Innovation import Innovation
from neat.NeuronGene import NeuronGene
from neat.LinkGene import LinkGene
from neat.InnovationType import InnovationType
from neat.NeuronType import NeuronType
from typing import *


class InnovationDB:
    """
    Class represents a database of innovations.
    """

    def __init__(self,
                 innovations = [],
                 next_neuron_id: int = 0,
                 next_innovation_num: int = 0):

        self.innovations = innovations
        self.next_neuron_id = next_neuron_id
        self.next_innovation_num = next_innovation_num


    @classmethod
    def from_genes(cls, link_genes: List[LinkGene], neuron_genes: List[NeuronGene]) -> 'InnovationDB':
        """
        Creates innovation database from neuron and link genes.

        :param link_genes:      List[LinkGene]      - link genes
        :param neuron_genes:    List[NeuronGene]    - neuron genes
        :return:                InnovationDB        - database of innovations
        """

        ret = cls()
        ret.next_neuron_id = 0
        ret.next_innovation_num = 0

        # add neurons
        for neuron_gene in neuron_genes:
            ret.innovations.append(Innovation.create_neuron_innovation(neuron_gene.neuron_id, -1, -1,
                                                                       ret.next_innovation_num,
                                                                       NeuronType.INPUT))
            ret.next_innovation_num += 1
            ret.next_neuron_id += 1

        # add links
        for link_gene in link_genes:
            innovation = Innovation.create_link_innovation(link_gene.from_neuron_id,
                                                           link_gene.to_neuron_id,
                                                           ret.next_innovation_num)
            ret.innovations.append(innovation)
            link_gene.innovation_id = innovation.innovation_id
            ret.next_innovation_num += 1

        return ret


    def check_link_innovation(self,
                              neuron_from_id: int,
                              neuron_to_id: int) -> int:
        """
        Checks if link innovation exists. Returns innovation id if it exists, or -1 otherwise.

        :param neuron_in_id:    int             - id of the input neuron
        :param neuron_out_id:   int             - id of the output neuron
        :return:                int             - If exists, id of the innovation, else -1
        """

        for innovation in self.innovations:
            if innovation.neuron_from_id == neuron_from_id and innovation.neuron_to_id == neuron_to_id \
                    and innovation.innovation_type == InnovationType.NEW_LINK:
                return innovation.innovation_id
        return -1


    def check_neuron_innovation(self,
                                neuron_id: int):
        """
        Checks if neuron innovation exists.
        :param neuron_id:
        :return: innovation id if it exists, -1 otherwise
        """
        for innovation in self.innovations:
            if innovation.neuron_id == neuron_id and innovation.innovation_type == InnovationType.NEW_NEURON:
                return innovation.innovation_id
        return -1


    def check_neuron_between_innovation(self,
                                        neuron_in_id: int,
                                        neuron_out_id: int):
        """
        Checks if neuron innovation between two other neurons exists.
        :param neuron_in_id:    int
        :param neuron_out_id:   int
        :return: innovation id if it exists, -1 otherwise
        """
        for innovation in self.innovations:
            if innovation.neuron_from_id == neuron_in_id and \
               innovation.neuron_to_id == neuron_out_id and \
               innovation.innovation_type == InnovationType.NEW_NEURON:
                return innovation.innovation_id
        return -1

    def create_link_innovation(self,
                               neuron_from_id: int,
                               neuron_to_id: int):
        """
        Creates innovation of type NEW_LINK.

        :param neuron_from_id:    int             - id of the input neuron
        :param neuron_to_id:   int             - id of the output neuron
        :return:
        """

        innovation = Innovation.create_link_innovation(neuron_from_id, neuron_to_id, self.next_innovation_num)

        self.innovations.append(innovation)
        self.next_innovation_num += 1
        return self.next_neuron_id - 1


    def create_neuron_innovation(self,
                                 from_neuron_id: int,
                                 to_neuron_id: int,
                                 neuron_type: NeuronType):
        """
        Creates a new neuron innovation.

        :param from_neuron_id:  int             - first neuron, where new one is between
        :param to_neuron_id:    int             - second neuron, where new one is between
        :param neuron_type:     NeuronType      - type of neuron
        :return:
        """

        innovation = Innovation.create_neuron_innovation(self.next_neuron_id, from_neuron_id, to_neuron_id,
                                                         self.next_innovation_num, neuron_type)
        self.innovations.append(innovation)
        self.next_innovation_num += 1
        self.next_neuron_id += 1

        return self.next_neuron_id - 1


    def create_neuron_from_id(self, neuron_id) -> NeuronGene:
        """
        Creates neuron with given id. If neuron already exists in
        the database of innovations, fetches it. Otherwise, creates
        new neuron with default values.

        :param neuron_id:   int         - id of the neuron
        :return:            NeuronGene
        """
        tmp = NeuronGene.constructor3(neuron_id)
        for innovation in self.innovations:
            if innovation.neuron_id == neuron_id:
                tmp.neuron_type = innovation.neuron_type
                tmp.neuron_id = neuron_id
                return tmp
        return tmp


    def next_number(self, num=0):
        """

        :param num:
        :return:
        """
        self.next_innovation_num += num
        return self.next_innovation_num


    def get_neuron_id(self, inv):
        """

        :param inv:
        :return:
        """
        return self.innovations[inv].neuron_id
