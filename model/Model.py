import tensorflow as tf
import numpy as np
import os

MODELS_PATH = "models"


class Model:
    def __init__(self, neurons, links, input_neurons_num=4):
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
        # list of all links in graph
        self.links = links
        # number of inputs
        self.input_neurons_num = input_neurons_num

        # ------- TensorFlow attributes -------
        # TensorFlow graph
        self.graph = None
        # inputs in graph
        self.inputs = {}
        # output from graph
        self.output = None

    def build_model(self):
        """
        Fill input_neurons, connections and weights.
        """
        for link in self.links:
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
        self.graph = tf.Graph()
        with self.graph.as_default():
            operations = {}
            # create Variables for input vertices
            for neuron_id in self.input_neurons:
                self.inputs[neuron_id] = tf.get_variable(name=str(neuron_id), shape=(),
                                                         initializer=tf.zeros_initializer)

            # create input & output vertices
            for neuron_id, input_neuron_ids in self.connections.items():
                # weights
                v_weights = tf.constant(self.weights[neuron_id])
                # input vertices
                v_inputs = []

                for input_neuron_id in input_neuron_ids:
                    if self.is_input_neuron(input_neuron_id):
                        vertex = self.inputs[input_neuron_id]
                    else:
                        vertex = operations[input_neuron_id]

                    v_inputs.append(vertex)
                # multiply weights and inputs
                mul = tf.multiply(v_inputs, v_weights, str(neuron_id))
                # sum multiplied values
                sum = tf.reduce_sum(mul, name='sum_' + str(neuron_id))
                # apply activation function
                if self.is_output_neuron(neuron_id):
                    activation = tf.sigmoid(sum, name="output")
                else:
                    activation = tf.nn.leaky_relu(sum, alpha=0.2, name="relu_" + str(neuron_id))

                operations[neuron_id] = activation
                if self.is_output_neuron(neuron_id):
                    self.output = activation
        return self.graph, self.inputs, self.output

    def build(self):
        self.build_model()
        self.build_graph()

    def is_output_neuron(self, neuron_id):
        """
        Check if 'neuron_id' neuron is output of graph.
        :param neuron_id:
        :return:
        """
        return self.neurons[self.input_neurons_num].neuron_id == neuron_id

    def is_input_neuron(self, neuron_id):
        """
        Check if 'neuron_id' neuron is input to graph.
        :param neuron_id: id of neuron
        :return:
        """
        for i in range(self.input_neurons_num):
            if self.neurons[i].neuron_id == neuron_id:
                return True
        return False

    def calculate_loss(self, X, y):
        """
        Predict category for each row of data set, compare it with truth category and return how good predictions are.
        Higher value means better graph.
        :param X: data set
        :param y: true categories
        :return:
        """
        probs = self.predict(X)

        num_examples = X.shape[0]

        sub = np.subtract(probs, y)
        sqr = np.square(sub)
        sm = np.sum(sqr)
        return 1 - sm / num_examples

    def feed(self, data):
        """
        Create feed_dict for session.
        :param data:
        :return:
        """
        feed = {}
        for i in range(len(self.inputs)):
            feed[self.inputs[i]] = data[i]
        return feed

    def predict(self, X):
        """
        Predict class for 'X'
        :param X: array of shape (n, self.input_neurons_num), n > 0; example: [[1,2,3]]; [[1,2,3],[1,2,3]]
        :return:
        """
        probs = []
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            for x in X:
                probs.append(sess.run(self.output, feed_dict=self.feed(x)))
        return probs

    def save_graph_summary(self):
        """
        Save the computation graph to a TensorBoard summary file.
        When file is saved, in a new terminal, launch TensorBoard with the following shell command:
            tensorboard --logdir .
        """
        writer = tf.summary.FileWriter('.')
        writer.add_graph(self.graph)

    def save_graph(self):
        """
        Save graph to file, so later can be loaded.
        :return:
        """
        with tf.Session(graph=self.graph) as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            save_path = saver.save(sess, os.path.join(MODELS_PATH, "model"))
            print("Model saved in path: %s" % save_path)

            with open("models/.model.inputs", "w") as file:
                for v in self.inputs.values():
                    file.write(v.name + "\n")
            with open("models/.model.output", "w") as file:
                file.write(self.output.name)
