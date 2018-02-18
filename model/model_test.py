from neat.Genome import Genome
from neat.LinkGene import LinkGene
from neat.NeuronGene import NeuronGene
from neat.NeuronType import NeuronType
from neat.graph import Graph

neurons = []
links = []

neurons.append(NeuronGene(0, NeuronType.INPUT))
neurons.append(NeuronGene(1, NeuronType.INPUT))
neurons.append(NeuronGene(2, NeuronType.INPUT))
neurons.append(NeuronGene(3, NeuronType.INPUT))
neurons.append(NeuronGene(4, NeuronType.OUTPUT))

neurons.append(NeuronGene(55, NeuronType.HIDDEN))
neurons.append(NeuronGene(56, NeuronType.HIDDEN))
neurons.append(NeuronGene(57, NeuronType.HIDDEN))
neurons.append(NeuronGene(60, NeuronType.HIDDEN))
neurons.append(NeuronGene(63, NeuronType.HIDDEN))

links.append(LinkGene(0, 4))
links.append(LinkGene(1, 55))
links.append(LinkGene(55, 4))
links.append(LinkGene(2, 63))
links.append(LinkGene(63, 57))
links.append(LinkGene(57, 4))
links.append(LinkGene(3, 55))
links.append(LinkGene(3, 56))
links.append(LinkGene(56, 60))
links.append(LinkGene(60, 4))
links.append(LinkGene(57, 56))
links.append(LinkGene(60, 56))


g = Genome.from_neurons_and_links(0, neurons, links, 4, 1)
graph = Graph.from_genome(g)
print(graph.is_cyclic_graph())
# model = g.create_phenotype()
# g.get_fitness()
# model.save_graph_summary()
