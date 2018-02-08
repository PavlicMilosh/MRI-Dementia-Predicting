from neat.LinkGene import LinkGene
from neat.ParentType import ParentType
from random import random
from neat.Genome import Genome
from neat.InnovationDB import InnovationDB


class Ga(object):

    def __init__(self,
                 population_size: int,
                 inputs: int,
                 outputs: int,
                 generation: int = 0,
                 innovation_db: InnovationDB = None,
                 next_genome_id: int = 0,
                 next_species_id: int = 0,
                 fittest_genome_id: int = 0,
                 best_ever_fitness: float = 0,
                 total_fitness_adj: float = 0,
                 avg_fitness_adj: float = 0):

        self.inputs = inputs
        self.outputs = outputs
        self.innovation_db = innovation_db

        # current generation
        self.generation = generation
        self.next_genome_id = next_genome_id
        self.next_species_id = next_species_id
        self.population_size = population_size

        self.fittest_genome_id = fittest_genome_id
        self.best_ever_fitness = best_ever_fitness

        self.total_fitness_adj = total_fitness_adj  # adjusted fitness scores
        self.avg_fitness_adj = avg_fitness_adj

        self.genomes = []       # current population
        self.best_genomes = []  # last generation bests
        self.species = []       # all species

        self.splits = []        # split depth

        for i in range(self.population_size):
            next_genome_id += 1
            self.genomes.append(Genome.from_inputs_outputs(next_genome_id, inputs, outputs))

        genome = Genome.from_inputs_outputs(1, inputs, outputs)
        self.innovationDB = InnovationDB.from_genes(genome.links, genome.neurons)


    # ======================================================================================================================
    # CROSSOVER
    # ======================================================================================================================

    def crossover(self, mother: Genome, father: Genome) -> Genome:

        best_parent = None

        # choose best parent
        if mother.fitness() == father.fitness():
            if mother.num_links() == father.num_links():
                best_parent = ParentType.MOTHER if random.uniform(0, 1) > 0.5 else ParentType.FATHER
            else:
                best_parent = ParentType.MOTHER if mother.num_links() < father.num_links() else ParentType.FATHER
        else:
            best_parent = ParentType.MOTHER if mother.fitness() > father.fitness() else ParentType.FATHER

        baby_neurons = []
        baby_links = []
        neuron_ids = []

        mother_it = iter(mother.start_of_links())
        father_it = iter(father.start_of_links())

        cur_mother = LinkGene.constructor()
        cur_father = LinkGene.constructor()

        selected_link = None

        # loop through genes while the end of both parent genes
        # isn't reached
        while cur_mother is not None and cur_father is not None:

            # if we reached the end of mother genes
            if cur_mother is None and cur_father is not None:
                if best_parent == ParentType.FATHER:
                    selected_link = cur_father
                cur_father = next(father_it, None)

            # if we reached the end of fathers genes
            elif cur_mother is not None and cur_father is None:
                if best_parent == ParentType.MOTHER:
                    selected_link = cur_mother
                cur_mother = next(mother_it, None)

            # if mothers innovation is less (older) than fathers
            elif cur_mother < cur_father:
                if best_parent == ParentType.MOTHER:
                    selected_link = cur_mother
                cur_mother = next(mother_it, None)

            # if fathers innovation is less (older) than mothers
            elif cur_father < cur_mother:
                if best_parent == ParentType.FATHER:
                    selected_link = cur_father
                cur_father = next(father_it, None)

            # if innovations are same
            elif cur_mother.innovation_id == cur_father.innovation_id:
                selected_link = cur_mother if random.uniform(0, 1) > 0.5 else cur_father
                cur_father = next(father_it, None)
                cur_mother = next(mother_it, None)

            if len(baby_links) == 0:
                baby_links.append(selected_link)
            elif baby_links[len(baby_links) - 1].innovation_id != selected_link.innovation_id:
                baby_links.append(selected_link)

            self.add_neuron_id(selected_link.from_neuron_id, neuron_ids)
            self.add_neuron_id(selected_link.to_neuron_id, neuron_ids)

        neuron_ids.sort()

        for n_id in neuron_ids:
            baby_neurons.append(self.innovation_db.create_neuron_from_id(n_id))

        baby_genome = Genome.from_neurons_and_links(self.next_genome_id,
                                                    baby_neurons,
                                                    baby_links,
                                                    mother.inputs,
                                                    mother.outputs)
        self.next_genome_id += 1

        return baby_genome


    def add_neuron_id(self, neuron_id: int, neuron_ids: list):
        for n in neuron_ids:
            if n.neuron_id == neuron_id:
                return
        neuron_ids.append(neuron_id)

