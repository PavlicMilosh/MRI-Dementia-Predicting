from data_preprocessing.preprocess_data import get_training_data, get_test_data
from neat.Constants import epsilon, max_generation, population_size
from neat.Ga import Ga


def evolve_networks(pop_size: int, num_inputs: int, num_outputs: int):
    ga = Ga(pop_size, num_inputs, num_outputs)

    phenotypes = ga.create_phenotypes()

    x_train, y_train = get_training_data()

    while True:
        fitness_scores = []
        for phenotype in phenotypes:
            fitness_scores.append(phenotype.calculate_loss(x_train, y_train))

        phenotypes = ga.epoch(fitness_scores)

        if ga.best_genomes[0].fitness > epsilon or ga.generation > max_generation:
            return ga.fittest_genome


if __name__ == '__main__':
    best_network = evolve_networks(population_size, 15, 1)
    model = best_network.phenotype.save_graph()
    # do evaluation with evaluation set
    x_test, y_test = get_test_data()
    fitness = model.calculate_loss(x_test, y_test)
