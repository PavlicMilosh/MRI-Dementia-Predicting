from neat.LinkGene import LinkGene
from neat.ParentType import ParentType
from random import random
from neat.Genome import Genome

# ======================================================================================================================
# CROSSOVER
# ======================================================================================================================


def crossover(self, mother: Genome, father: Genome) -> Genome:

    best_parent = None

    # choose best parent
    if mother.fitness() == father.fitness():
        if mother.num_genes() == father.num_genes():
            best_parent = ParentType.MOTHER if random.uniform(0, 1) > 0.5 else ParentType.FATHER
        else:
            best_parent = ParentType.MOTHER if mother.num_genes() < father.num_genes() else ParentType.FATHER
    else:
        best_parent = ParentType.MOTHER if mother.fitness() > father.fitness() else ParentType.FATHER

    baby_neurons = []
    baby_genes = []
    vec_neurons = []

    mother_it = iter(mother.start_of_links())
    father_it = iter(father.start_of_links())

    cur_mother = LinkGene.constructor()
    cur_father = LinkGene.constructor()

    selected_gene = None

    # loop through genes while the end of both parent genes
    # haven't been reached
    while cur_mother is not None and cur_father is not None:

        # if we reached the end of mother genes
        if cur_mother is None and cur_father is not None:
            if best_parent == ParentType.FATHER:
                selected_gene = cur_father
            cur_father = next(father_it, None)

        # if we reached the end of fathers genes
        elif cur_mother is not None and cur_father is None:
            if best_parent == ParentType.MOTHER:
                selected_gene = cur_mother
            cur_mother = next(mother_it, None)

        # if mothers innovation is less (older) than fathers
        elif cur_mother.innovation_id < cur_father.innovation_id:
            if best_parent == ParentType.MOTHER:
                selected_gene = cur_mother
            cur_mother = next(mother_it, None)

        # if fathers innovation is less (older) than mothers
        elif cur_father.innovation_id < cur_mother.innovation_id:
            if best_parent == ParentType.FATHER:
                selected_gene = cur_father
            cur_father = next(father_it, None)

        # if innovations are same
        elif cur_mother.innovation_id == cur_father.innovation_id:
            selected_gene = cur_mother if random.uniform(0, 1) > 0.5 else cur_father
            cur_father = next(father_it, None)
            cur_mother = next(mother_it, None)

        if len(baby_genes) == 0:
            baby_genes.append(selected_gene)
        elif baby_genes[len(baby_genes) - 1].innovation_id != selected_gene.innovation_id:
            baby_genes.append(selected_gene)

    return Genome()

