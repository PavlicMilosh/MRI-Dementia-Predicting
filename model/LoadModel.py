import tensorflow as tf
import os


class LoadModel:
    """
    Class which represent loaded model from file.
    Use it like this:
        model = LoadModel()
        prediction = model.predict([[1, 2, 3]])
    """

    def __init__(self, path="models"):
        self._models_path = path
        self._graph = None
        self._inputs = {}
        self._output = None
        self._restore_graph()

    def _restore_graph(self):
        """
        Restore saved graph from file.
        :return:
        """
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(os.path.join(self._models_path, 'model.meta'))
            saver.restore(sess, tf.train.latest_checkpoint(os.path.join('.', self._models_path)))
            self._graph = tf.get_default_graph()

            with open(os.path.join(self._models_path, '.model.inputs')) as f:
                for line in f.readlines():
                    line = line.strip()
                    self._inputs[line] = self._graph.get_tensor_by_name(line)

            with open(os.path.join(self._models_path, '.model.output')) as f:
                line = f.readline().strip()
            self._output = self._graph.get_tensor_by_name(line)
            # return graph, inputs, output

    def predict(self, X):
        """
        Predict class for 'X'
        :param X: array of shape (n, self.input_neurons_num), n > 0; example: [[1,2,3]]; [[1,2,3],[1,2,3]]
        :return:
        """
        probs = []
        with tf.Session(graph=self._graph) as sess:
            sess.run(tf.global_variables_initializer())
            for x in X:
                probs.append(sess.run(self._output, feed_dict=self._feed(x)))
        return probs

    def _feed(self, data):
        """
        Create feed_dict for session.
        :param data:
        :return:
        """
        feed = {}
        for i in range(len(self._inputs)):
            input_name = str(i) + ":0"
            feed[self._inputs[input_name]] = data[i]
        return feed
