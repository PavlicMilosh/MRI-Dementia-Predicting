
from data_preprocessing.preprocess_data import get_test_data, get_training_data
from model.LoadModel import LoadModel
from neat.Constants import epsilon, max_generation, population_size
from neat.Ga import Ga
import numpy as np


def evolve_networks(pop_size: int, num_inputs: int, num_outputs: int):
    ga = Ga(pop_size, num_inputs, num_outputs)

    ga.create_phenotypes()
    for genome in ga.genomes:
        genome.get_fitness()

    while True:

        print("=======================================================================================================")
        print("Epoch:                  [ " + str(ga.generation) + " ]")
        print("Best ever fitness:      [ " + str("{:6.5f}").format(ga.best_ever_fitness) + " ]")
        if ga.fittest_genome is not None:
            print("Fittest genome:         [ " + str(ga.fittest_genome.genome_id) + " ]")
            print("Fittest genome fitness: [ " + str("{:6.5f}").format(ga.fittest_genome.fitness) + " ]")
        print("=======================================================================================================")

        # fitness_scores = []
        # for genotype in ga.genomes:
        #     fitness_scores.append(genotype.get_fitness())

        ga.epoch()

        if ga.best_ever_fitness > epsilon or ga.generation > max_generation:
            return ga.fittest_genome


def main():
    best_network = evolve_networks(population_size, 5, 1)
    model = best_network.create_phenotype()
    model.save_graph()
    model.save_graph_summary()

    # do evaluation with evaluation set

    loaded_model = LoadModel()

    x_test, y_test = get_test_data()
    x_train, y_train = get_training_data()

    y_predict = loaded_model.predict(x_test)
    y_train_predict = loaded_model.predict(x_train)
    train_accuracy = 1 - np.sum(np.abs(np.subtract(y_train_predict, y_train))) / x_train.shape[0]
    accuracy = 1 - np.sum(np.abs(np.subtract(y_predict, y_test))) / x_test.shape[0]
    print(train_accuracy)
    print(accuracy)

    # model.calculate_loss(x_test, y_test)


if __name__ == '__main__':
    main()
