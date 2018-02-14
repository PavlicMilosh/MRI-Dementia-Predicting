import tensorflow as tf


class Model:
    def __init__(self, neurons, input_neurons_num):
        """
        Constructor.
        :param neurons: array list of neurons
        :param input_neurons_num: number of input neurons
        """
        # neuron_id as key and weights entering in it as value
        self.weights = {}
        # neuron_id as key and neurons entering in it as value
        self.connections = {}
        # set of inputs to graph
        self.input_neurons = set()
        # list of all neurons in graph; list is supposed to be sorted in way that
        # fist 'input_neurons_num' nodes are input, then goes one output, then hidden nodes
        self.neurons = neurons
        # number of inputs
        self.input_neurons_num = input_neurons_num

    def build_model(self, links):
        """
        Fill input_neurons, connections and weights.
        :param links: list of links where each link has:
            1. from_neuron_id
            2. to_neuron_id
            3. weight
        """
        for link in links:
            # if from neuron is input to graph, add it to input_neurons set
            if self.is_input_neuron(link.from_neuron_id):
                self.input_neurons.add(link.from_neuron_id)
            # add weight to neuron
            if link.to_neuron_id not in self.weights:
                self.weights[link.to_neuron_id] = []
            self.weights[link.to_neuron_id].append(link.weight)
            # add input to neuron
            if link.to_neuron_id not in self.connections:
                self.connections[link.to_neuron_id] = []
            self.connections[link.to_neuron_id].append(link.from_neuron_id)

    def build_graph(self):
        """
        Create TensorFlow Computational Graph based on model.
        :return graph: constructed TF Graph
        :return inputs: inputs to graph
        :return output: output of graph
        """
        graph = tf.Graph()
        with graph.as_default():
            operations = {}
            # inputs in graph
            inputs = {}
            # create Variables for input vertices
            for neuron_id in self.input_neurons:
                inputs[neuron_id] = tf.get_variable(name=str(neuron_id), shape=(), initializer=tf.zeros_initializer)

            # create input & output vertices
            for neuron_id, input_neuron_ids in self.connections.items():
                # weights
                v_weights = tf.constant(self.weights[neuron_id])
                # input vertices
                v_inputs = []

                for input_neuron_id in input_neuron_ids:
                    if self.is_input_neuron(input_neuron_id):
                        vertex = inputs[input_neuron_id]
                    else:
                        vertex = operations[input_neuron_id]

                    v_inputs.append(vertex)
                # multiply weights and inputs
                mul = tf.multiply(v_inputs, v_weights, str(neuron_id))
                # sum multiplied values
                sum = tf.reduce_sum(mul, name='sum_' + str(neuron_id))
                # apply activation function
                if self.is_output_neuron(neuron_id):
                    activation = tf.sigmoid(sum, name="sigmoid" + str(neuron_id))
                else:
                    activation = tf.nn.leaky_relu(sum, alpha=0.2, name="relu_" + str(neuron_id))

                operations[neuron_id] = activation
                if self.is_output_neuron(neuron_id):
                    output = activation
        return graph, inputs, output

    def is_output_neuron(self, neuron_id):
        """
        Check if 'neuron_id' neuron is output of graph.
        :param neuron_id:
        :return:
        """
        return self.neurons[self.input_neurons_num] == neuron_id

    def is_input_neuron(self, neuron_id):
        """
        Check if 'neuron_id' neuron is input to graph.
        :param neuron_id: id of neuron
        :return:
        """
        for i in range(self.input_neurons_num):
            if self.neurons[i] == neuron_id:
                return True
        return False

    def calculate_loss(self, X, y):
        pass

    def predict(self, X):
        pass

    def save_graph(self):
        pass

    @staticmethod
    def save_graph_summary(graph):
        """
        Save the computation graph to a TensorBoard summary file.
        When file is saved, in a new terminal, launch TensorBoard with the following shell command:
            tensorboard --logdir .

        :param graph: tf.Graph
        """
        writer = tf.summary.FileWriter('.')
        writer.add_graph(graph)

    @staticmethod
    def create_feed(self, X):
        pass
