from random import uniform, randint
from math import sqrt
from typing import List

from data_preprocessing.preprocess_data import get_training_data
from neat.Constants import excess_multiplier, disjoint_multiplier, matched_multiplier
from neat.InnovationDB import InnovationDB
from neat.LinkGene import LinkGene
from neat.NeuronGene import NeuronGene
from neat.NeuronType import NeuronType
from model import Model
from neat.graph import Graph


class Genome(object):
    """
    Class represents genotype.
    """

    # ==================================================================================================================
    # CONSTRUCTORS
    # ==================================================================================================================

    def __init__(self,
                 genome_id: int = 0,
                 neurons: List[NeuronGene] = [],
                 links: List[LinkGene] = [],
                 phenotype = None,
                 fitness: float = 0.0,
                 adjusted_fitness: float = 0.0,
                 amount_to_spawn: int = 0,
                 num_inputs: int = 0,
                 num_outputs: int = 0,
                 species: int = 0,
                 inputs: int = 0,
                 outputs: int = 0,
                 depth: int = 0):

        self.genome_id = genome_id
        self.neurons = neurons
        self.links = links
        self.model = None
        self.phenotype = phenotype
        self.fitness = fitness
        self.adjusted_fitness = adjusted_fitness
        self.amount_to_spawn = amount_to_spawn
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.species = species
        self.inputs = inputs
        self.outputs = outputs
        self.depth = depth

    @classmethod
    def from_neurons_and_links(cls,
                               genome_id: int,
                               neurons: List[NeuronGene],
                               links: List[LinkGene],
                               inputs: int,
                               outputs: int) -> 'Genome':
        """
        Creates object of class Genome from list of neuron and
        link genes and number of inputs and outputs.

        :param genome_id:   int                 - id
        :param neurons:     List[NeuronGene]    - neurons
        :param links:       List[LinkGene]      - links
        :param inputs:      int                 - number of inputs
        :param outputs:     int                 - number of outputs
        :return: ret        Genome
        """
        ret = cls()
        ret.genome_id = genome_id
        ret.phenotype = None
        ret.links = links
        ret.neurons = neurons
        ret.amount_to_spawn = 0
        ret.fitness = 0
        ret.adjusted_fitness = 0
        ret.inputs = inputs
        ret.outputs = outputs
        return ret


    @classmethod
    def from_inputs_outputs(cls, genome_id: int, inputs: int, outputs: int, innovation_db: InnovationDB):
        """
        Creates object of class Genome from number of inputs and outputs.

        :param genome_id: int               - id
        :param inputs:    int               - number of inputs
        :param outputs:   int               - number of outputs
        :param innovation_db: InnovationDB  - database of all innovations
        :return:          Genome
        """

        ret = cls()
        ret.inputs = inputs
        ret.outputs = outputs
        ret.genome_id = genome_id
        ret.neurons = []
        ret.links = []

        for i in range(inputs):
            neuron = innovation_db.create_neuron_from_id(i)
            neuron.neuron_type = NeuronType.INPUT

            if innovation_db.check_neuron_innovation(neuron.neuron_id) == -1:
                innovation_db.create_neuron_innovation(-1, -1, neuron.neuron_type)
            neuron.innovation_id = innovation_db.check_neuron_innovation(neuron.neuron_id)
            ret.neurons.append(neuron)
        for o in range(outputs):
            neuron = innovation_db.create_neuron_from_id(o + inputs)
            neuron.neuron_type = NeuronType.OUTPUT
            ret.neurons.append(neuron)
            innovation_db.create_neuron_innovation(-1, -1, neuron.neuron_type)
            neuron.innovation_id = innovation_db.check_neuron_innovation(neuron.innovation_id)
            output_neuron_id = neuron.neuron_id

            for i in range(inputs):

                input_neuron_id = ret.neurons[i].neuron_id

                innovation_id = innovation_db.check_link_innovation(input_neuron_id,
                                                                    output_neuron_id)
                if innovation_id == -1:
                    innovation_id = innovation_db.next_innovation_num
                    innovation_db.create_link_innovation(input_neuron_id,
                                                         output_neuron_id)

                ret.links.append(LinkGene(input_neuron_id, output_neuron_id,
                                          1.0, True, False, innovation_id))

        return ret


    # ==================================================================================================================
    # MUTATIONS
    # ==================================================================================================================


    def add_link(self,
                 mutation_rate: float,
                 chance_of_looped: float,
                 innovation_db: InnovationDB,
                 num_tries_to_find_loop: int,
                 num_tries_to_add_link: int):
        """
        Tries to add a new link mutation to genotype.
        If mutation doesn't exist in the database of
        innovations, adds mutation as innovation, else
        gets existing innovation.

        :param mutation_rate:           float
        :param chance_of_looped:        float
        :param innovation_db:           InnovationDB
        :param num_tries_to_find_loop:  int
        :param num_tries_to_add_link:   int
        """
        if uniform(0, 1) > mutation_rate:
            return

        neuron1_id = -1
        neuron2_id = -1

        recurrent = False

        if uniform(0, 1) < chance_of_looped:
            for i in range(num_tries_to_find_loop):

                neuron_pos = randint(self.inputs, len(self.neurons) - 1)

                if not self.neurons[neuron_pos].recurrent \
                        and self.neurons[neuron_pos].neuron_type != NeuronType.BIAS \
                        and self.neurons[neuron_pos].neuron_type != NeuronType.INPUT:
                    neuron1_id = neuron2_id = self.neurons[neuron_pos].neuron_id
                    self.neurons[neuron_pos].recurrent = True

                    recurrent = True
                    break

        else:
            for i in range(num_tries_to_add_link):

                neuron1_id = self.neurons[randint(0, len(self.neurons) - 1)].neuron_id
                neuron2_id = self.neurons[randint(self.inputs, len(self.neurons) - 1)].neuron_id

                # first neuron must not be output
                if self.neurons[self.get_element_pos(neuron1_id)].neuron_type == NeuronType.OUTPUT:
                    continue

                # second neuron must not be input or bias
                if self.neurons[self.get_element_pos(neuron2_id)].neuron_type == NeuronType.INPUT and \
                   self.neurons[self.get_element_pos(neuron2_id)].neuron_type == NeuronType.BIAS:
                    continue

                if self.duplicate_link(neuron1_id, neuron2_id) or neuron1_id == neuron2_id:
                    neuron1_id = -1
                    neuron2_id = -1
                else:
                    break

        if neuron1_id < 0 or neuron2_id < 0:
            return

        graph = Graph.from_genome(self)
        graph.insert_edge(graph.find_vertex(neuron1_id), graph.find_vertex(neuron2_id))
        if graph.is_cyclic_graph():
            return

        innovation_id = innovation_db.check_link_innovation(neuron1_id, neuron2_id)

        # if self.neurons[self.get_element_pos(neuron1_id)].split_y > \
        #         self.neurons[self.get_element_pos(neuron2_id)].split_y:
        #     recurrent = True

        if innovation_id < 0:
            innovation_db.create_link_innovation(neuron1_id, neuron2_id)
            innovation_id = innovation_db.next_number() - 1

        gene = LinkGene.constructor1(neuron1_id, neuron2_id, True, innovation_id, uniform(-1, 1), recurrent)
        self.links.append(gene)


    def add_neuron(self, mutation_rate: float, innovation_db: InnovationDB, num_tries_to_find_old_link: int):

        """
        Tries to add a new neuron mutation to genotype.
        If mutation doesn't exist in the database of
        innovations, adds mutation as innovation, else
        gets existing innovation.


        :param mutation_rate:               float
        :param innovation_db:               InnovationDB
        :param num_tries_to_find_old_link:  int
        :return:
        """

        if uniform(0, 1) > mutation_rate:
            return

        old_link_found = False
        chosen_link_index = 0

        size_threshold = self.inputs + self.outputs + 5

        if len(self.links) > size_threshold:

            for i in range(num_tries_to_find_old_link):

                chosen_link_index = randint(0, len(self.links) - 1 - int(sqrt(len(self.links))))
                from_neuron = self.links[chosen_link_index].from_neuron_id

                if self.links[chosen_link_index].enabled and \
                   not self.links[chosen_link_index].recurrent and \
                   self.neurons[self.get_element_pos(from_neuron)].neuron_type != NeuronType.BIAS:
                    old_link_found = True
                    break

            if not old_link_found:
                return

        else:

            while not old_link_found:
                chosen_link_index = randint(0, len(self.links) - 1)
                from_neuron = self.links[chosen_link_index].from_neuron_id
                if self.links[chosen_link_index].enabled:
                    if not self.links[chosen_link_index].recurrent:
                        if self.neurons[self.get_element_pos(from_neuron)].neuron_type != NeuronType.BIAS:
                            old_link_found = True

        self.links[chosen_link_index].enabled = False

        original_weight = self.links[chosen_link_index].weight
        from_neuron_id = self.links[chosen_link_index].from_neuron_id
        to_neuron_id = self.links[chosen_link_index].to_neuron_id

        # new_depth = (self.neurons[self.get_element_pos(from_neuron)].split_y
        #              + self.neurons[self.get_element_pos(to_neuron)].split_y) / 2
        # new_width = (self.neurons[self.get_element_pos(from_neuron)].split_x
        #              + self.neurons[self.get_element_pos(to_neuron)].split_x) / 2

        innovation_id = innovation_db.check_neuron_between_innovation(from_neuron_id, to_neuron_id)

        # if innovation was found, check if it already exists
        if innovation_id > 0:
            neuron_id = innovation_db.get_neuron_id(innovation_id)
            if self.already_have_this_neuron_id(neuron_id):
                innovation_id = -1

        # if innovation doesn't exists
        if innovation_id < 0:

            new_neuron_id = innovation_db.create_neuron_innovation(from_neuron_id, to_neuron_id, NeuronType.HIDDEN)

            self.neurons.append(NeuronGene.constructor1(NeuronType.HIDDEN, new_neuron_id, innovation_id))

            # create first link between first neuron and new neuron
            link1_id = innovation_db.next_number()
            innovation_db.create_link_innovation(from_neuron_id, new_neuron_id)
            link1 = LinkGene.constructor1(from_neuron_id, new_neuron_id, True, link1_id, 1.0)
            self.links.append(link1)

            # create second link between new neuron and second neuron
            link2_id = innovation_db.next_number()
            innovation_db.create_link_innovation(new_neuron_id, to_neuron_id)
            link2 = LinkGene.constructor1(new_neuron_id, to_neuron_id, True, link2_id, original_weight)
            self.links.append(link2)

        else:

            new_neuron_id = innovation_db.get_neuron_id(innovation_id)
            link1_id = innovation_db.check_link_innovation(from_neuron_id, new_neuron_id)
            link2_id = innovation_db.check_link_innovation(new_neuron_id, to_neuron_id)

            # if one of links doesn't exist in innovation database
            if (link1_id < 0) or (link2_id < 0):
                self.links[chosen_link_index].enabled = True
                return

            link1 = LinkGene.constructor1(from_neuron_id, new_neuron_id, True, link1_id, 1.0)
            link2 = LinkGene.constructor1(new_neuron_id, to_neuron_id, True, link2_id, original_weight)

            self.links.append(link1)
            self.links.append(link2)

            self.neurons.append(NeuronGene.constructor1(NeuronType.HIDDEN, new_neuron_id, innovation_id))


    def mutate_weights(self,
                       mutation_rate: float,
                       new_mutation_probability: float,
                       max_perturbation: float):
        """
        Tries to mutate weights of links.

        :param mutation_rate:               float
        :param new_mutation_probability:    float
        :param max_perturbation:            float
        """
        for idx, val in enumerate(self.links):
            if uniform(0, 1) < mutation_rate:
                if uniform(0, 1) < new_mutation_probability:
                    self.links[idx].weight = uniform(-1, 1)
                else:
                    self.links[idx].weight = uniform(-1, 1) * max_perturbation

    def mutate_activation_response(self,
                                   mutation_rate: float,
                                   max_perturbation: float):
        """
        Tries to mutate neuron activation function.

        :param mutation_rate:       float
        :param max_perturbation:    float
        """
        for idx, val in enumerate(self.neurons):
            if uniform(0, 1) < mutation_rate:
                self.neurons[idx].activation_response += uniform(-1, 1) * max_perturbation

    # ==================================================================================================================
    # PHENOTYPE
    # ==================================================================================================================

    def create_phenotype(self):
        self.model = Model.Model(self.neurons, self.links, self.inputs)
        self.model.build()
        return self.model

    def delete_phenotype(self):
        self.phenotype = None

    # ==================================================================================================================
    # HELPER METHODS
    # ==================================================================================================================

    def duplicate_link(self, neuron_from_id: int, neuron_to_id: int) -> bool:
        """
        Checks if link between given neurons (represented through id)
        already exists.

        :param neuron_to_id:
        :param neuron_from_id:
        :return:                bool    - returns True if exists, else False
        """
        for l in self.links:
            if l.from_neuron_id == neuron_from_id and l.to_neuron_id == neuron_to_id:
                return True
        return False


    def already_have_this_neuron_id(self, neuron_id: int) -> bool:
        """
        Checks if genome already contains neuron represented with
        given id.

        :param neuron_id:    int
        :return:             bool    - returns True if contains, else False
        """
        for n in self.neurons:
            if n.neuron_id == neuron_id:
                return True
        return False


    def sort_link(self):
        """
        Sorts genome links by innovation_id.
        """
        self.links.sort()

    # ==================================================================================================================
    # ACCESSOR METHODS
    # ==================================================================================================================

    def num_links(self):
        return len(self.links)


    def num_neurons(self):
        return len(self.neurons)


    def start_of_links(self):
        return iter(self.links)


    def get_fitness(self):
        X, y = get_training_data()
        if self.model is None:
            raise AttributeError("Error. Phenotype is not created")
        else:
            self.fitness = self.model.calculate_loss(X, y)
            return self.model.calculate_loss(X, y)


    def get_element_pos(self, neuron_id: int) -> int:
        """
        Finds position (indx) of neuron by given neuron id.

        :param neuron_id: int   - id of neuron
        :return:          int   - position if found, -1 if not
        """
        for i in range(len(self.neurons)):
            if self.neurons[i].neuron_id == neuron_id:
                return i
        return -1


    # ==================================================================================================================
    # SCORE
    # ==================================================================================================================

    def get_compatibility_score(self, other: 'Genome'):
        disjoint = 0
        excess = 0
        matched = 0
        weight_difference = 0

        i1 = 0
        i2 = 0

        while(i1 < len(self.links) - 1) or (i2 < len(other.links) - 1):
            if i1 == len(self.links) - 1:
                i2 += 1
                excess += 1
                continue

            if i2 == len(other.links) - 1:
                i1 += 1
                excess += 1
                continue

            link1 = self.links[i1]
            link2 = self.links[i2]

            if link1.innovation_id == link2.innovation_id:
                i1 += 1
                i2 += 1
                matched += 1

                weight_difference = abs(link1.weight - link2.weight)

            if link1 < link2:
                disjoint += 1
                i1 += 1

            if link2 < link1:
                disjoint += 1
                i2 += 1

        longest = max(self.num_links(), other.num_links())

        if matched == 0:
            matched = 1

        return (excess_multiplier * excess / float(longest)) +\
               (disjoint_multiplier * disjoint / float(longest)) +\
               (matched_multiplier * weight_difference / matched)

    # ==================================================================================================================
    # OPERATOR OVERLOAD
    # ==================================================================================================================

    def __lt__(self, other):
        return self.fitness < other.fitness
