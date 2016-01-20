import lasagne
import theano.tensor as tensor
import theano
import numpy as np
from theano.compile.nanguardmode import NanGuardMode
from lasagne.nonlinearities import tanh, rectify, leaky_rectify
from lasagne.updates import sgd, nesterov_momentum
from lasagne.objectives import squared_error
from lasagne.objectives import squared_error
from lasagne.regularization import regularize_layer_params
import lasagne.layers as ls

def double_tanh(x):
    return 2*tanh(x)

def quadruple_tanh(x):
    return 4*tanh(x)

class MLPEvaluator:
    def __init__(self, state_format, actions_number, batch_size, network_args, gamma=0.99, updates=sgd, learning_rate = 0.01, regularization = None):
        self._loss_history = []
        self._misc_state_included = (state_format[1] > 0)
        self._gamma = gamma
        if self._misc_state_included:
            self._misc_inputs = tensor.matrix('misc_inputs')
            self._misc_input_shape = (1, state_format[1])
            self._misc_buffer = np.ndarray((batch_size, state_format[1]), dtype=np.float32)
            self._misc_buffer2 = np.ndarray((batch_size, state_format[1]), dtype=np.float32)
        else:
            self._misc_input_shape = None

        self._targets = tensor.matrix('targets')
        self._image_inputs = tensor.tensor4('image_inputs')

        network_image_input_shape = list(state_format[0])
        network_image_input_shape.insert(0, None)

        self._image_input_shape = list(network_image_input_shape)
        self._image_input_shape[0] = batch_size

        # create buffers for batch learning
        self._input_image_buffer = np.ndarray(self._image_input_shape, dtype=np.float32)
        self._input_image_buffer2 = np.ndarray(self._image_input_shape, dtype=np.float32)

        # save it for the evaluation reshape
        self._image_input_shape[0] = 1

        network_args["img_input_shape"] = network_image_input_shape
        network_args["misc_shape"] = self._misc_input_shape
        network_args["output_size"] = actions_number

        self._initialize_network(**network_args)
        print "Network initialized."
        self._compile(updates, learning_rate, regularization)

    def _compile(self, updates, learning_rate, regularization ):
        
        predictions = ls.get_output(self._network)
        regularization_term = 0.0
        if regularization:
            for method, coefficient in regularization:
                regularization_term += coefficient * regularize_layer_params(self._network, method)

        loss = squared_error(predictions, self._targets).mean()
        regularized_loss = loss + regularization_term
        params = ls.get_all_params(self._network, trainable=True)
        updates = updates(regularized_loss, params, learning_rate=learning_rate)


        print "Compiling Theano functions ..."
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
        print "Theano functions compiled."

    def _initialize_network(self, img_input_shape, misc_shape, output_size, hidden_units=[500],
                            hidden_layers=1, hidden_nonlin=leaky_rectify, output_nonlin=double_tanh, updates=sgd):
        print "Initializing MLP network..."
        # image input layer
        network = ls.InputLayer(shape=img_input_shape, input_var=self._image_inputs)
        # hidden layers
        for i in range(hidden_layers):
            network = ls.DenseLayer(network, hidden_units[i], nonlinearity=hidden_nonlin)
        if self._misc_state_included:
            # misc input layer
            misc_input_layer = ls.InputLayer(shape=misc_shape, input_var=self._misc_inputs)
            # merge layer
            network = ls.ConcatLayer([network, misc_input_layer])

        # output layer
        network = ls.DenseLayer(network, output_size, nonlinearity = output_nonlin)
        self._network = network

        self._img_i_s = img_input_shape
        self._hid_uns = hidden_units

    def learn(self, transitions):

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
        
        # substitute expected values for chosen actions
        for i, q in zip(range(len(transitions)), q2):
            target[i][transitions[i][1]] = transitions[i][3]
            if transitions[i][2] is not None:
                target[i][transitions[i][1]] += self._gamma * q
        

        if self._misc_state_included:
            loss = self._learn(self._input_image_buffer, self._misc_buffer, target)
        else:
            loss = self._learn(self._input_image_buffer, target)
        self._loss_history.append(loss)

    def best_action(self, state):
        if self._misc_state_included:
            qvals = self._evaluate(state[0].reshape(self._image_input_shape), state[1].reshape(self._misc_input_shape))
            a = np.argmax(qvals)
        else:
            qvals = self._evaluate(state[0].reshape(self._image_input_shape))
            a = np.argmax(qvals)
        return a
        
    def get_mean_loss(self, clear = True):
        m = np.mean(self._loss_history)
        self._loss_history = []
        return m

    def get_network(self):
        return self._network

class CNNEvaluator(MLPEvaluator):
    def __init__(self, **kwargs):
        MLPEvaluator.__init__(self, **kwargs)

    def _initialize_network(self, img_input_shape, misc_shape, output_size, conv_layers=2, num_filters=[32, 32],
                            filter_size=[(5, 5), (5, 5)], hidden_units=[256], pool_size=[(2, 2), (2, 2)],
                            hidden_layers=1, conv_nonlin=rectify,
                            hidden_nonlin=leaky_rectify, output_nonlin=double_tanh):

        print "Initializing CNN ..."
        # image input layer
        network = ls.InputLayer(shape=img_input_shape, input_var=self._image_inputs)

        # convolution and pooling layers
        for i in range(conv_layers):
            network = ls.Conv2DLayer(network, num_filters=num_filters[i], filter_size=filter_size[i],
                                                 nonlinearity=conv_nonlin)
            network = ls.MaxPool2DLayer(network, pool_size=pool_size[i])
        # dense layers
        for i in range(hidden_layers):
            network = ls.DenseLayer(network, hidden_units[i], nonlinearity=hidden_nonlin)
        
        if self._misc_state_included:
            # misc input layer
            misc_input_layer = ls.InputLayer(shape=misc_shape, input_var=self._misc_inputs)
            # merge layer
            network = ls.ConcatLayer([network, misc_input_layer])

        # output layer
        network = ls.DenseLayer(network, output_size, nonlinearity=output_nonlin)
        self._network = network
