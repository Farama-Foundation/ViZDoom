import lasagne
import theano.tensor as tensor
import theano
import numpy as np
from theano.compile.nanguardmode import NanGuardMode


class MLPEvaluator:
    def __init__(self, state_format, actions_number, batch_size, network_args, gamma=0.99):

        self._misc_state_included = (state_format[1] > 0)
        self._gamma = gamma

        if self._misc_state_included:
            self._misc_inputs = tensor.matrix('misc_inputs')
            misc_input_shape = (None, state_format[1])
            self._misc_input_shape = (1, state_format[1])
            self._misc_buffer = np.ndarray((batch_size, state_format[1]), dtype=np.float32)
            self._misc_buffer2 = np.ndarray((batch_size, state_format[1]), dtype=np.float32)
        else:
            misc_input_shape = None
        image_dimensions = len(state_format[0])

        self._targets = tensor.matrix('targets')
        self._image_inputs = tensor.tensor4('image_inputs')

        network_image_input_shape = list(state_format[0])
        network_image_input_shape.insert(0, None)
        self._image_input_shape = list(network_image_input_shape)
        self._image_input_shape[0] = batch_size

        # create buffers for batch learning
        self._input_image_buffer = np.ndarray(self._image_input_shape, dtype=np.float32)
        self._input_image_buffer2 = np.ndarray(self._image_input_shape, dtype=np.float32)
        self._expected_buffer = np.ndarray([batch_size], dtype=np.float32)

        # save it for the evaluation reshape
        self._image_input_shape[0] = 1

        network_args["img_shape"] = network_image_input_shape
        network_args["misc_shape"] = misc_input_shape
        network_args["output_size"] = actions_number
        self._initialize_network(**network_args)

    def _initialize_network(self, img_shape, misc_shape, output_size, hidden_units=[500], learning_rate=0.01,
                            hidden_layers=1, hidden_nonlin=lasagne.nonlinearities.tanh, updates=lasagne.updates.sgd):
        print "Initializing MLP network..."
        # image input layer
        network = lasagne.layers.InputLayer(shape=img_shape, input_var=self._image_inputs)
        # hidden layers
        for i in range(hidden_layers):
            network = lasagne.layers.DenseLayer(network, hidden_units[i], nonlinearity=hidden_nonlin)
        if self._misc_state_included:
            # misc input layer
            misc_input_layer = lasagne.layers.InputLayer(shape=misc_shape, input_var=self._misc_inputs)
            # merge layer
            network = lasagne.layers.ConcatLayer([network, misc_input_layer])

        # output layer
        network = lasagne.layers.DenseLayer(network, output_size, nonlinearity=None)
        self._network = network

        predictions = lasagne.layers.get_output(network)
        loss = lasagne.objectives.squared_error(predictions, self._targets).mean()
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = updates(loss, params, learning_rate=learning_rate)

        # mode = NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
        mode = None
        if self._misc_state_included:
            self._learn = theano.function([self._image_inputs, self._misc_inputs, self._targets], loss, updates=updates,
                                          mode=mode, name="learn_fn")
            self._evaluate = theano.function([self._image_inputs, self._misc_inputs], predictions, mode=mode,
                                             name="eval_fn")
        else:
            self._learn = theano.function([self._image_inputs, self._targets], loss, updates=updates)
            self._evaluate = theano.function([self._image_inputs], predictions)

    def learn_one(self, s, a, s2, r):

        if self._misc_state_included:
            s[0] = s[0].reshape(self._image_input_shape)
            s[1] = s[1].reshape(self._misc_input_shape)
            target = self._evaluate(s[0], s[1])
            # find best q values for s2
            if s2 is not None:
                s2[0] = s2[0].reshape(self._image_input_shape)
                s2[1] = s2[1].reshape(self._misc_input_shape)
                q2 = np.max(self._evaluate(s2[0], s2[1]))
        else:
            s = s[0].reshape(self._image_input_shape)
            target = self._evaluate(s)
            # find best q values for s2
            if s2 is not None:
                s2 = s2[0].reshape(self._image_input_shape)
                q2 = np.max(self._evaluate(s2))

        # set expected output as the reward got from the transition
        expected_q = r

        if s2 is not None:
            expected_q += self._gamma * q2

        target[0, a] = expected_q

        if self._misc_state_included:
            self._learn(s[0], s[1], target)
        else:
            self._learn(s, target)

    def learn(self, transitions):

        # TODO:
        # change internal representation of transitions so that it would return
        # ready ndarrays
        # prepare the batch

        if self._misc_state_included:
            for i, trans in zip(range(len(transitions)), transitions):
                self._input_image_buffer[i] = trans[0][0]
                self._misc_buffer[i] = trans[0][1]
                # if it's the terminal state just ignore
                if trans[2] is not None:
                    self._input_image_buffer2[i] = trans[2][0]
                    self._misc_buffer2[i] = trans[2][1]

            target = self._evaluate(self._input_image_buffer, self._misc_buffer)
            # find best q values for s2
            q2 = np.max(self._evaluate(self._input_image_buffer2, self._misc_buffer2), axis=1)

        else:
            for i, trans in zip(range(len(transitions)), transitions):

                self._input_image_buffer[i] = trans[0][0]
                # if it's the terminal state just ignore
                if trans[2] is not None:
                    self._input_image_buffer2[i] = trans[2][0]

            target = self._evaluate(self._input_image_buffer)
            # find best q values for s2
            q2 = np.max(self._evaluate(self._input_image_buffer2), axis=1)

        # set expected output as the reward got from the transition
        for i, trans in zip(range(len(transitions)), transitions):
            self._expected_buffer[i] = trans[3]

        # substitute expected values for chosen actions
        for i, q in zip(range(len(transitions)), q2):
            if transitions[i][2] is not None:
                self._expected_buffer[i] += self._gamma * q
            target[i][transitions[i][1]] = self._expected_buffer[i]

        if self._misc_state_included:
            self._learn(self._input_image_buffer, self._misc_buffer, target)
        else:
            self._learn(self._input_image_buffer, target)

    def best_action(self, state):
        if self._misc_state_included:
            a = np.argmax(
                self._evaluate(state[0].reshape(self._image_input_shape), state[1].reshape(self._misc_input_shape)))
        else:
            a = np.argmax(self._evaluate(state[0].reshape(self._image_input_shape)))
        return a


class CNNEvaluator(MLPEvaluator):
    def __init__(self, **kwargs):
        MLPEvaluator.__init__(self, **kwargs)

    def _initialize_network(self, img_shape, misc_shape, output_size, conv_layers=2, num_filters=[32, 32],
                            filter_size=[(5, 5), (5, 5)], hidden_units=[256], pool_size=[(2, 2), (2, 2)],
                            learning_rate=0.01, hidden_layers=1, conv_nonlin=lasagne.nonlinearities.rectify,
                            hidden_nonlin=lasagne.nonlinearities.tanh, updates=lasagne.updates.sgd):

        print "Initializing CNN ..."
        # image input layer
        network = lasagne.layers.InputLayer(shape=img_shape, input_var=self._image_inputs)

        # convolution and pooling layers
        for i in range(conv_layers):
            network = lasagne.layers.Conv2DLayer(network, num_filters=num_filters[i], filter_size=filter_size[i],
                                                 nonlinearity=conv_nonlin)
            network = lasagne.layers.MaxPool2DLayer(network, pool_size=pool_size[i])
        # dense layers
        for i in range(hidden_layers):
            network = lasagne.layers.DenseLayer(network, hidden_units[i], nonlinearity=hidden_nonlin)
        if self._misc_state_included:
            # misc input layer
            misc_input_layer = lasagne.layers.InputLayer(shape=misc_shape, input_var=self._misc_inputs)
            # merge layer
            network = lasagne.layers.ConcatLayer([network, misc_input_layer])

        # output layer
        network = lasagne.layers.DenseLayer(network, output_size, nonlinearity=None)
        self._network = network

        predictions = lasagne.layers.get_output(network)
        loss = lasagne.objectives.squared_error(predictions, self._targets).mean()
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = updates(loss, params, learning_rate=learning_rate)

        # mode = NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
        mode = None
        if self._misc_state_included:
            self._learn = theano.function([self._image_inputs, self._misc_inputs, self._targets], loss, updates=updates,
                                          mode=mode, name="learn_fn")
            self._evaluate = theano.function([self._image_inputs, self._misc_inputs], predictions, mode=mode,
                                             name="eval_fn")
        else:
            self._learn = theano.function([self._image_inputs, self._targets], loss, updates=updates)
            self._evaluate = theano.function([self._image_inputs], predictions)
