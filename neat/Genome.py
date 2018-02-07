import random
from math import sqrt

from neat.InnovationDB import InnovationDB
from neat.LinkGene import LinkGene
from neat.NeuronGene import NeuronGene
from neat.InovationType import InnovationType
from neat.NeuronType import NeuronType


class Genome(object):

    # ==================================================================================================================
    # CONSTRUCTORS
    # ==================================================================================================================

    def __init__(self,
                 genome_id: int,
                 neurons: list,
                 links: list,
                 phenotype,
                 fitness: float,
                 adjusted_fitness: float,
                 amount_to_spawn: int,
                 num_inputs: int,
                 num_outputs: int,
                 species: int,
                 inputs: int,
                 outputs: int):

        self.genome_id = genome_id
        self.neurons = neurons
        self.links = links
        self.phenotype = phenotype
        self.fitness = fitness
        self.adjusted_fitness = adjusted_fitness
        self.amount_to_spawn = amount_to_spawn
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.species = species
        self.inputs = inputs
        self.outputs = outputs


    @classmethod
    def from_inputs_outputs(cls, inputs: int, outputs: int):

        ret = cls()
        ret.neurons = []
        ret.links = []

        for i in range(inputs):
            ret.neurons.append(NeuronGene(i, NeuronType.INPUT, False, 0, 0, 0))

        for i in range(outputs):
            ret.neurons.append(NeuronGene(i + inputs, NeuronType.OUTPUT, False, 0, 1, 1))
            for j in range(inputs):
                ret.links.append(LinkGene(ret.neurons[j], ret.neurons[i + inputs], 1, True, False, 0))

        return ret


    @classmethod
    def from_links_neurons(cls,
                           neurons: list,
                           links: list,
                           inputs: int,
                           outputs: int):
        ret = cls()
        ret.neurons = neurons
        ret.links = links
        ret.inputs = inputs
        ret.outputs = outputs
        return ret


    def duplicate_link(self, neuron_in_id: int, neuron_out_id: int):
        for l in self.links:
            if l.from_neuron_id == neuron_in_id and l.to_neuron_id == neuron_out_id:
                return True
        return False


    def already_have_this_neuron_id(self, neuron_id: int):
        for n in self.neurons:
            if n.neuron_id == neuron_id:
                return True
        return False


    # given a neuron id this function just finds its position in
    # m_vecNeurons
    def get_element_pos(self, neuron_id: int) -> int:
        pass


    def create_phenotype(self):
        pass


    # ==================================================================================================================
    # MUTATIONS
    # ==================================================================================================================

    # ADD LINK =========================================================================================================

    def add_link(self, mutation_rate: float, chance_of_looped: float, innovation_db: InnovationDB,
                 num_tries_to_find_loop: int, num_tries_to_add_link: int):

        if random.uniform(0, 1) > mutation_rate:
            return

        neuron1_id = -1
        neuron2_id = -1

        recurrent = False

        if random.uniform(0, 1) > chance_of_looped:
            for i in range(num_tries_to_find_loop):
                neuron_pos = random.randint(self.inputs + 1, len(self.neurons) - 1)
                if not self.neurons[neuron_pos].recurrent and self.neurons[neuron_pos].neuron_type != NeuronType.BIAS \
                        and self.neurons[neuron_pos].neuron_type != NeuronType.INPUT:
                    neuron1_id = neuron2_id = self.neurons[neuron_pos].neuron_id
                    self.neurons[neuron_pos].recurrent = True

                    recurrent = True
                    break

        else:
            for i in range(num_tries_to_add_link):
                neuron1_id = self.neurons[random.randint(0, len(self.neurons) - 1)].neuron_id
                neuron2_id = self.neurons[random.randint(self.inputs + 1, len(self.neurons) - 1)].id
                # second neuron must not be input or bias
                if self.neurons[self.get_element_pos(neuron2_id)].neuron_type != NeuronType.INPUT \
                        and self.neurons[self.get_element_pos(neuron2_id)].neuronType != NeuronType.BIAS:
                    continue

                if self.duplicate_link(neuron1_id, neuron2_id) or neuron1_id == neuron2_id:
                    neuron1_id = -1
                    neuron2_id = -1
                else:
                    break
        if neuron1_id < 0 or neuron2_id < 0:
            return

        innovation_id = innovation_db.check_innovation(neuron1_id, neuron2_id, InnovationType.NEW_LINK)

        if self.neurons[self.get_element_pos(neuron1_id)].split_y > \
                self.neurons[self.get_element_pos(neuron2_id)].split_y:
            recurrent = True

        if innovation_id < 0:
            innovation_db.create_link_innovation(neuron1_id, neuron2_id, InnovationType.NEW_LINK)
            innovation_id = innovation_db.next_number() - 1
            gene = LinkGene.constructor(neuron1_id, neuron2_id, True, innovation_id, random.uniform(-1, 1), recurrent)
            self.links.append(gene)
        else:
            gene = LinkGene.constructor(neuron1_id, neuron2_id, True, innovation_id, random.uniform(-1, 1), recurrent)
            self.links.append(gene)


    # ADD NEURON =======================================================================================================

    def add_neuron(self, mutation_rate: float, innovation_db: InnovationDB, num_tries_to_find_old_link: int):

        if random.uniform(0, 1) > mutation_rate:
            return

        old_link_found = False
        chosen_link_index = 0

        size_threshold = self.inputs + self.outputs + 5

        if len(self.links) > size_threshold:
            for i in range(num_tries_to_find_old_link):
                chosen_link_index = random.randint(0, len(self.links) - 1 - int(sqrt(len(self.links))))

                from_neuron = self.links[chosen_link_index].from_neuron

                if self.links[chosen_link_index].enabled and not self.links[chosen_link_index].recurrent \
                        and self.neurons[self.get_element_pos(from_neuron)].neuron_type != NeuronType.BIAS:
                    old_link_found = True
                    break

            if not old_link_found:
                return

        else:
            while not old_link_found:
                chosen_link_index = random.randint(0, len(self.links) - 1)

                from_neuron = self.links[chosen_link_index].from_neuron

                if self.links[chosen_link_index].enabled and not self.links[chosen_link_index].recurrent \
                        and self.neurons[self.get_element_pos(from_neuron)].neuron_type != NeuronType.BIAS:
                    old_link_found = True

        self.links[chosen_link_index].enabled = False

        original_weight = self.links[chosen_link_index].weight

        from_neuron = self.links[chosen_link_index].from_neuron
        to_neuron = self.links[chosen_link_index].to_neuron

        new_depth = (self.neurons[self.get_element_pos(from_neuron)].split_y
                     + self.neurons[self.get_element_pos(to_neuron)].split_y) / 2
        new_width = (self.neurons[self.get_element_pos(from_neuron)].split_x
                     + self.neurons[self.get_element_pos(to_neuron)].split_x) / 2

        innovation_id = innovation_db.check_innovation(from_neuron, to_neuron, InnovationType.NEW_NEURON)

        if innovation_id > 0:
            neuron_id = innovation_db.get_neuron_id(innovation_id)
            if self.already_have_this_neuron_id(neuron_id):
                innovation_id = -1

        if innovation_id < 0:
            new_neuron_id = innovation_db.create_neuron_innovation(from_neuron, to_neuron,
                                                                   InnovationType.NEW_NEURON, NeuronType.HIDDEN,
                                                                   new_width, new_depth)

            self.neurons.append(NeuronGene.constructor(NeuronType.HIDDEN, new_neuron_id, new_depth, new_width))

            link1_id = innovation_db.next_number()
            innovation_db.create_link_innovation(from_neuron, new_neuron_id, InnovationType.NEW_LINK)

            link1 = LinkGene.constructor(from_neuron, new_neuron_id, True, link1_id, 1., False)
            self.links.append(link1)

            link2_id = innovation_db.next_number()
            innovation_db.create_link_innovation(new_neuron_id, to_neuron, InnovationType.NEW_LINK)

            link2 = LinkGene.constructor(new_neuron_id, from_neuron, True, link2_id, original_weight, False)
            self.links.append(link2)

        else:
            new_neuron_id = innovation_db.get_neuron_id(innovation_id)

            link1_id = innovation_db.check_innovation(from_neuron, new_neuron_id, InnovationType.NEW_LINK)
            link2_id = innovation_db.check_innovation(new_neuron_id, to_neuron, InnovationType.NEW_LINK)

            if (link1_id < 0) or (link2_id < 0):
                return

            link1 = LinkGene.constructor(from_neuron, new_neuron_id, True, link1_id, 1., False)
            link2 = LinkGene.constructor(new_neuron_id, to_neuron, True, link2_id, original_weight, False)

            self.links.append(link1)
            self.links.append(link2)

            new_neuron = NeuronGene.constructor(NeuronType.HIDDEN, new_neuron_id, new_depth, new_width)
            self.neurons.append(new_neuron)


    # WEIGHT MUTATION ==================================================================================================

    def mutate_weights(self,
                       mutation_rate: float,
                       new_mutation_probability: float,
                       max_perturbation: float):

        for idx, val in enumerate(self.links):
            if random.uniform(0, 1) < mutation_rate:
                if random.uniform(0, 1) < new_mutation_probability:
                    self.links[idx].weight = random.uniform(-1, 1)
                else:
                    self.links[idx].weight = random.uniform(-1, 1) * max_perturbation


    # ACTIVATION MUTATION ==============================================================================================

    def mutate_activation_response(self,
                                   mutation_rate: float,
                                   max_perturbation: float):
        for idx, val in enumerate(self.neurons):
            if random.uniform(0, 1) < mutation_rate:
                self.neurons[idx].activation_response += random.uniform(-1, 1) * max_perturbation


    def __lt__(self, other):
        return self.fitness < other.fitness


    # ==================================================================================================================
    # FITNESS
    # ==================================================================================================================

    def fitness(self):
        pass


    # ==================================================================================================================
    # ACCESSOR METHODS
    # ==================================================================================================================

    def num_genes(self):
        return len(self.links)

    def num_neurons(self):
        return len(self.neurons)

    def start_of_links(self):
        return next(iter(self.links))

    def end_of_link(self):
        return None