from random import random, randint
from typing import List

from neat import Constants
from neat.Constants import *
from neat.Genome import Genome
from neat.InnovationDB import InnovationDB
from neat.LinkGene import LinkGene
from neat.ParentType import ParentType
from neat.Species import Species
from neat.SplitDepth import SplitDepth


class Ga(object):
    """
    Class represents genetic algorithm
    """

    def __init__(self,
                 population_size: int,
                 inputs: int,
                 outputs: int,
                 generation: int                = 0,
                 innovation_db: InnovationDB    = None,
                 next_genome_id: int            = 0,
                 next_species_id: int           = 0,
                 fittest_genome: Genome         = None,
                 best_ever_fitness: float       = 0.0,
                 total_fitness_adj: float       = 0.0,
                 avg_fitness_adj: float         = 0.0):

        self.inputs = inputs
        self.outputs = outputs
        self.innovation_db = innovation_db

        # current generation
        self.generation = generation
        self.next_genome_id = next_genome_id
        self.next_species_id = next_species_id
        self.population_size = population_size

        self.fittest_genome = fittest_genome
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


    def crossover(self, mother: Genome, father: Genome) -> Genome:
        """
        Performs a crossover between two genomes, returns a new combination (baby genome)

        :param mother:  Genome      - one parent for crossover
        :param father:  Genome      - other parent for crossover
        :return:        Genome      - baby genome
        """

        best_parent = None

        # choose the best parent
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


    def add_neuron_id(self, neuron_id: int, neuron_ids: List[int]):
        """
        Adds neuron id if it doesn't exists in the list.

        :param neuron_id:   int         - Id of neuron to be added
        :param neuron_ids:  List[int]   -List of neurons
        :return:
        """
        for n in neuron_ids:
            if n == neuron_id:
                return
        neuron_ids.append(neuron_id)


    def create_phenotypes(self):
        """
        Creates and returns phenotypes from the genomes.

        :return:
        """
        networks = []
        for genome in self.genomes:
            self.calculate_net_depth(genome)

            network = genome.create_phenotype()

            networks.append(network)

        return networks


    def calculate_net_depth(self, genome: Genome):
        """
        Calculates network depth of a given genome.

        :param genome: Genome
        :return:
        """
        max_so_far = 0

        for nd in range(genome.num_neurons()):
            for split in self.splits:
                if genome.split_y(nd) > split.val:
                    max_so_far = split.depth

        genome.depth = max_so_far + 2


    def epoch(self, fitness_scores: List[float]):
        """
        Performs one epoch of genetic algorithm and returns a list of new phenotypes.

        :param fitness_scores:  List[float]           - scores
        :return:                List[Phenotype]       - phenotypes
        """
        print("Epoch: " + str(self.generation) + ", Best ever fitness: " + str(self.best_ever_fitness))

        if len(fitness_scores) != len(self.genomes):
            return None

        self.reset_and_kill()

        for gen in range(len(self.genomes)):
            self.genomes[gen].fitness = fitness_scores[gen]

        self.sort_and_record()

        self.speciate_and_calculate_spawn_levels()

        new_population = []
        spawned_so_far = 0

        for spc in range(len(self.species)):
            if spawned_so_far < population_size:
                num_to_spawn = int(round(self.species[spc].spawns_required))
                chosen_best_yet = False

                for i in range(num_to_spawn):

                    if not chosen_best_yet:
                        baby = self.species[spc].leader
                        chosen_best_yet = True

                    else:
                        if self.species[spc].number_of_members() == 1:
                            baby = self.species[spc].spawn()

                        else:
                            g1 = self.species[spc].spawn()

                            if random.randrange(0, 1) < crossover_rate:
                                g2 = self.species[spc].spawn()

                                attempts = 5

                                while (g1.genome_id == g2.genome_id) and attempts > 0:
                                    g2 = self.species[spc].spawn()

                                if g1.genome_id != g2.genome_id:
                                    baby = self.crossover(g1, g2)

                            else:
                                baby = g1

                        self.next_genome_id += 1

                        baby.genome_id = self.next_genome_id

                        if baby.num_neurons() < max_permitted_neurons:
                            baby.add_neuron(chance_to_add_neuron, self.innovation_db, num_tries_to_find_old_link)

                        baby.add_link(chance_to_add_link, chance_to_add_recurrent_link, self.innovation_db,
                                      num_tries_to_find_loop, num_tries_to_add_link)

                        baby.mutate_weights(mutation_rate, probability_of_weight_replacement, max_weight_perturbation)

                        baby.mutate_activation_response(activation_mutation_rate, max_activation_perturbation)

                    # baby.sort_genes()

                    new_population.append(baby)

                    spawned_so_far += 1

                    if spawned_so_far == population_size:
                        break

        if spawned_so_far < population_size:
            rqd = population_size - spawned_so_far

            for i in range(rqd):
                new_population.append(self.tournament_selection(round(population_size / 5)))

        self.genomes = new_population

        new_phenotypes = []

        for genotype in self.genomes:
            self.calculate_net_depth(self.genomes[genotype])
            phenotype = genotype.create_phenotype()
            new_phenotypes.append(phenotype)

        self.generation += 1

        return new_phenotypes


    def reset_and_kill(self):
        """
        Resets some values ready for the next epoch,
        and kill all phenotypes and poorly performing species.
        """

        for species in self.species:
            species.purge()

            if species.gens_no_improvement() > gens_allowed_with_no_improvement \
                    and species.best_fitness < self.best_ever_fitness:
                self.species.remove(species)

            for genome in self.genomes:
                genome.delete_phenotype()


    def sort_and_record(self):
        """
        Sorts the population by fitness descending and keeps record of the best n genomes
        and updates any fitness statistics accordingly.
        """
        self.genomes.sort()
        self.genomes.reverse()

        if self.genomes[0].fitness > self.best_ever_fitness:
            self.best_ever_fitness = self.genomes[0].fitness
            self.fittest_genome = self.genomes[0]

        self.best_genomes.clear()
        for index in range(best_sweepers_num):
            self.best_genomes.append(self.genomes[index])


    def speciate_and_calculate_spawn_levels(self):
        """
        Places individuals into their respecting species by calculating compatibility
        with other members of the population and niching accordingly.
        Adjusting the fitness scores of each individual by species age and by sharing
        and determines how many offsprings each individual should spawn.
        """

        self.adjust_compatibility_threshold()

        for genome in self.genomes:
            for species in self.species:

                compatibility = genome.get_compatibility_score(species.leader)

                if compatibility < compatibility_threshold:
                    species.add_member(genome)
                    genome.species = species
                    break
            else:
                self.species.append(Species.init(genome, self.next_species_id))
                self.next_species_id += 1

        self.adjust_species_fitness()

        for genome in self.genomes:
            self.total_fitness_adj += genome.adjusted_fitness

        self.avg_fitness_adj = self.total_fitness_adj / len(self.genomes)

        for genome in self.genomes:
            to_spawn = genome.adjusted_fitness / self.avg_fitness_adj
            genome.amount_to_spawn = to_spawn

        for species in self.species:
            species.calculate_spawn_amount()


    def adjust_compatibility_threshold(self):
        """
        Automatically adjusts the compatibility threshold in an attempt
        to keep the number of species below the maximum.
        """

        if max_number_of_species < 1:
            return

        threshold_increment = 0.01

        if len(self.species) > max_number_of_species:
            Constants.compatibility_threshold += threshold_increment

        elif len(self.species) < 2:
            Constants.compatibility_threshold -= threshold_increment


    def adjust_species_fitness(self):
        """
        Iterates through each species and calls adjust_fitness for each species.
        """
        for species in self.species:
            species.adjust_fitness()


    def tournament_selection(self, num_comparisons: int) -> Genome:
        """

        :param num_comparisons: int     -
        :return:                Genome  - chosen genome
        """
        best_fitness_so_far = 0
        chosen = 0

        for i in range(num_comparisons):
            this_try = randint(0, len(self.genomes) - 1)

            if self.genomes[this_try].fitness > best_fitness_so_far:
                chosen = this_try
                best_fitness_so_far = self.genomes[this_try].fitness

        return self.genomes[chosen]


    def split(self, low: float, high: float, depth: float) -> List[SplitDepth]:
        """

        :param low:     float               -
        :param high:    float               -
        :param depth:   float               -
        :return:        List[SplitDepth]    -
        """
        splits = []
        span = high - low

        splits.append(SplitDepth(low + span / 2, depth + 1))

        if depth > 6:
            return splits
        else:
            self.split(low, low + span / 2, depth + 1)
            self.split(low + span / 2, high, depth + 1)

            return splits


    def get_best_phenotypes_from_last_generation(self):
        """

        :return:
        """
        brains = []

        for genome in self.best_genomes:
            self.calculate_net_depth(genome)

            brains.append(genome.create_phenotype())

        return brains
