import random
from typing import List
from math import ceil
from neat.Constants import young_age_threshold, young_fitness_bonus, old_age_threshold, old_age_penalty, survival_rate
from neat.Genome import Genome


class Species:
    """
    Class represents species.
    """

    # ==================================================================================================================
    # CONSTRUCTORS
    # ==================================================================================================================

    def __init__(self,
                 members: List[Genome] = [],
                 leader: Genome = None,
                 species_id: int = 0,
                 best_fitness: float = 0.0,
                 gens_no_improvement: int = 0,
                 age: int = 0,
                 spawns_required: float = 0.0):

        self.members = members
        self.leader = leader
        self.species_id = species_id
        self.best_fitness = best_fitness
        self.gens_no_improvement = gens_no_improvement
        self.age = age
        self.spawns_required = spawns_required


    @classmethod
    def init(cls, first: Genome, species_id: int):
        """
        Creates species

        :param first:       Genome  -
        :param species_id:  int     - id of the species
        :return:
        """
        ret = cls()
        ret.species_id = species_id
        ret.best_fitness = first.fitness
        ret.gens_no_improvement = 0
        ret.age = 0
        ret.leader = first
        ret.spawns_required = 0
        return ret


    # ==================================================================================================================
    # METHODS
    # ==================================================================================================================


    def add_member(self, new: Genome):
        """
        Adds new member to species.

        :param new: Genome  - new member
        """
        if new.fitness > self.best_fitness:
            self.best_fitness = new.fitness
            self.gens_no_improvement = 0
            self.leader = new

        self.members.append(new)


    def purge(self):
        """
        Purges species, deletes all members.
        """
        del self.members[:]

        self.age += 1
        self.gens_no_improvement += 1
        self.spawns_required = 0


    def adjust_fitness(self):
        """
        Adjusts fitness function.
        """
        total = 0
        for genome in self.members:
            fitness = genome.fitness
            if self.age < young_age_threshold:
                fitness *= young_fitness_bonus

            if self.age > old_age_threshold:
                fitness *= old_age_penalty

            total += fitness
            adjusted_fitness = fitness/len(self.members)
            genome.adjusted_fitness = adjusted_fitness


    def calculate_spawn_amount(self):
        """
        Calculates number of genomes to be spawned.
        """
        for genome in self.members:
            self.spawns_required += genome.amount_to_spawn


    def spawn(self) -> Genome:
        """
        Spawns new genome.

        :return: Genome     - baby genome
        """
        if len(self.members) == 1:
            baby = self.members[0]

        else:
            index = random.randint(0, ceil((survival_rate * len(self.members)) + 1))
            baby = self.members[index]

        return baby


    # ==================================================================================================================
    # ACCESSOR METHODS
    # ==================================================================================================================

    def number_of_members(self):
        return len(self.members)


    def leader_fitness(self):
        return self.leader.fitness


    def __lt__(self, other):
        return self.best_fitness < other.best_fitness





