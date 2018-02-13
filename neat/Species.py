import random
from typing import List

from neat.Constants import young_age_threshold, young_fitness_bonus, old_age_threshold, old_age_penalty, survival_rate
from neat.Genome import Genome


class Species:

    def __init__(self, members: List[Genome], leader: Genome, species_id: int, best_fitness: float,
                 gens_no_improvement: int, age: int, spawns_required: float):
        self.members = members
        self.leader = leader
        self.species_id = species_id
        self.best_fitness = best_fitness
        self.gens_no_improvement = gens_no_improvement
        self.age = age
        self.spawns_required = spawns_required

    @classmethod
    def init(cls, first: Genome, species_id: int):
        ret = cls()
        ret.species_id = species_id
        ret.best_fitness = first.fitness
        ret.gens_no_improvement = 0
        ret.age = 0
        ret.leader = first
        ret.spawns_required = 0
        return ret


    def add_member(self, new: Genome):
        if new.fitness > self.best_fitness:
            self.best_fitness = new.fitness
            self.gens_no_improvement = 0
            self.leader = new

        self.members.append(new)


    def purge(self):
        del self.members[:]

        self.age += 1
        self.gens_no_improvement += 1
        self.spawns_required = 0


    def adjust_fitness(self):
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
        for genome in self.members:
            self.spawns_required += genome.amount_to_spawn


    def spawn(self):
        if len(self.members) == 1:
            baby = self.members[0]

        else:
            index = random.randint(0, (survival_rate * len(self.members)) + 1)
            baby = self.members[index]

        return baby


    def number_of_members(self):
        return len(self.members)


    def leader_fitness(self):
        return self.leader.fitness


    def __lt__(self, other):
        return self.best_fitness < other.best_fitness





