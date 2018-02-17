import profile
import re

from data_preprocessing.preprocess_data import get_training_data, get_test_data
from model.LoadModel import LoadModel
from neat.Constants import epsilon, max_generation, population_size
from neat.Ga import Ga


def evolve_networks(pop_size: int, num_inputs: int, num_outputs: int):
    ga = Ga(pop_size, num_inputs, num_outputs)

    ga.create_phenotypes()

    while True:

        print("Epoch: [" + str(ga.generation) + "], Best ever fitness: [" + str(ga.best_ever_fitness) + "]")

        fitness_scores = []
        for genotype in ga.genomes:
            fitness_scores.append(genotype.get_fitness())

        ga.epoch(fitness_scores)

        if ga.best_genomes[0].fitness > epsilon or ga.generation > max_generation:
            return ga.fittest_genome


def main():
    best_network = evolve_networks(population_size, 4, 1)
    model = best_network.create_phenotype()
    model.save_graph()
    model.save_graph_summary()
    # do evaluation with evaluation set
    x_test, y_test = get_test_data()
    fitness = model.calculate_loss(x_test, y_test)
    print(fitness)


if __name__ == '__main__':
    main()
