from neat.Innovation import Innovation
from neat.util import InnovationType, NeuronType


class InnovationDB:

    def __init__(self, innovations):
        self.innovations = innovations


    def check_innovation(self, neuron1_id: int, neuron2_id: int, innovation_type: InnovationType) -> int:
        for i in self.innovations:
            if i.neuron_in == neuron1_id and i.neuron_out == neuron2_id and i.innovation_type == innovation_type:
                return i.innovation_id
        else:
            return -1


    def create_new_innovation(self, neuron1_id: int, neuron2_id: int, innovation_type: InnovationType):
        if self.check_innovation(neuron1_id, neuron2_id, innovation_type) == -1:
            self.innovations.append(Innovation(self.next_number(), InnovationType.NEW_LINK, neuron1_id, neuron2_id, \
                                               -1, NeuronType.NONE))
        else:
            return


    def next_number(self) -> int:
        return self.innovations[-1].innovation_id + 1
