import random

from neat.LinkGene import LinkGene
from neat.NeuronGene import NeuronGene
from neat.util import NeuronType, InnovationType


class Genome:

    def __init__(self, inputs: int, outputs: int):
        self.neurons = []
        self.links = []
        for i in range(inputs):
            self.neurons.append(NeuronGene(i, NeuronType.INPUT, False, 0, 0, 0))

        for i in range(outputs):
            self.neurons.append(NeuronGene(i + inputs, NeuronType.OUTPUT, False, 0, 1, 1))
            for j in range(inputs):
                self.links.append(LinkGene(self.neurons[j], self.neurons[i + inputs], 1, True, False, 0))

    def __init__(self, genome_id: int, neurons: list, links: list, phenotype, fitness: float, adjusted_fitness: float,
                 amount_to_spawn: int, num_inputs: int, num_outputs: int, species: int):
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

    def __init__(self, neurons: list, links: list, inputs: int, outputs: int):
        self.neurons = neurons
        self.links = links
        self.inputs = inputs
        self.outputs = outputs

    def duplicate_link(self, neuron_in, neuron_out):
        for link in self.links:
            if link.from_neuron == neuron_in and link.to_neuron == neuron_out:
                return True
        return False

    def already_have_this_neuron_id(self, neuron_id):
        for neuron in self.neurons:
            if neuron.neuron_id == neuron_id:
                return True
        return False

    # given a neuron id this function just finds its position in
    # m_vecNeurons
    def get_element_pos(self, neuron_id: int) -> int:
        pass

    def create_phenotype(self):
        pass

    def add_link(self, mutation_rate: float, chance_of_looped: float, innovation,
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
                if self.neurons[self.get_element_pos(neuron2_id)].neuron_type > 2:
                    continue

                if self.duplicate_link(neuron1_id, neuron2_id) or neuron1_id == neuron2_id:
                    neuron1_id = -1
                    neuron2_id = -1
                else:
                    break
        if neuron1_id < 0 or neuron2_id < 0:
            return

        id = innovation.check_innovation(neuron1_id, neuron2_id, InnovationType.NEW_LINK)

        if self.neurons[self.get_element_pos(neuron1_id)].split_y > \
                self.neurons[self.get_element_pos(neuron2_id)].split_y:
            recurrent = True

        if id < 0:
            innovation.create_new_innovation(neuron1_id, neuron2_id, InnovationType.NEW_LINK)
            id = innovation.next_number() - 1
            gene = LinkGene(neuron1_id, neuron2_id, True, id, random.uniform(-1, 1))
            self.links.append(gene)
        else:
            gene = LinkGene(neuron1_id, neuron2_id, True, id, random.uniform(-1, 1), recurrent)
            self.links.append(gene)

    def add_neuron(self):
        pass

    def __lt__(self, other):
        return self.fitness < other.fitness