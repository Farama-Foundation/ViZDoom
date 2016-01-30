import lasagne
import theano.tensor as tensor
import theano
import numpy as np
from theano.compile.nanguardmode import NanGuardMode
from lasagne.nonlinearities import tanh, rectify, leaky_rectify
from lasagne.updates import sgd, nesterov_momentum, norm_constraint
from lasagne.objectives import squared_error
from lasagne.objectives import squared_error
from lasagne.regularization import regularize_layer_params
import lasagne.layers as ls
from time import time

def relu_weights_initializer(alpha = 0.01):
    return lasagne.init.GlorotNormal(gain=np.sqrt(2/(1+alpha**2)))
    
class MLPEvaluator:
    def __init__(self, state_format, actions_number, network_args=dict(), gamma=0.99, updates=sgd, learning_rate = 0.01, regularization = None):

        self._loss_history = []
        self._misc_state_included = (state_format["s_misc"] > 0)
        self._gamma = gamma
        if self._misc_state_included:
            self._misc_inputs = tensor.matrix('misc_inputs')
            self._misc_len = state_format["s_misc"]
        else:
            self._misc_len = None

        self._targets = tensor.matrix('targets')
        self._image_inputs = tensor.tensor4('image_inputs')

        network_image_input_shape = list(state_format["s_img"])
        network_image_input_shape.insert(0, None)

        # save it for the evaluation reshape
        self._single_image_input_shape = list(network_image_input_shape)
        self._single_image_input_shape[0] = 1

        
        network_args["img_input_shape"] = network_image_input_shape
        network_args["misc_len"] = self._misc_len
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
        updates = updates(regularized_loss, params, learning_rate)


        print "Compiling Theano functions ..."
        
        # TODO find out why this causes problems with misc vector
        #mode = NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
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

    def _initialize_network(self, img_input_shape, misc_len, output_size, hidden_units=[500],
                            hidden_layers=1, hidden_nonlin=leaky_rectify, output_nonlin=tanh, updates=sgd):
        print "Initializing MLP network..."
        # image input layer
        network = ls.InputLayer(shape=img_input_shape, input_var=self._image_inputs)
        # hidden layers
        for i in range(hidden_layers):
            network = ls.DenseLayer(network, hidden_units[i], nonlinearity=hidden_nonlin, W=weights_initializer())
        
        # misc layer and merge with rest of the network
        if self._misc_state_included:
            # misc input layer
            misc_input_layer = ls.InputLayer(shape=(None, misc_len), input_var=self._misc_inputs)
            # merge layer
            network = ls.ConcatLayer([network, misc_input_layer])

        # output layer
        network = ls.DenseLayer(network, output_size, nonlinearity = output_nonlin)
        self._network = network

        self._img_i_s = img_input_shape
        self._hid_uns = hidden_units

    def learn(self, transitions):
        ## TODO get rid of second forward pass
        # Learning approximation: Q = r + terminal * 
        X = transitions["s1_img"]
        X2 = transitions["s2_img"]
        if self._misc_state_included:
            X_misc = transitions["s1_misc"]
            X2_misc = transitions["s2_misc"]
            Y =  self._evaluate(X, X_misc)
            Q2 = self._gamma*np.max(self._evaluate(X2,X2_misc),axis=1)
        else:
            Y =  self._evaluate(X)
            Q2 = self._gamma*np.max(self._evaluate(X2 ),axis=1) 

        for row,a,target in zip(Y,transitions["a"],transitions["r"] +(-transitions["terminal"]*Q2)):
            row[a] = target

        if self._misc_state_included:
            loss = self._learn(X, X_misc, Y)
        else:
            loss = self._learn(X, Y)
            
        self._loss_history.append(loss)


    def best_action(self, state):
        if self._misc_state_included:
            qvals = self._evaluate(state[0].reshape(self._single_image_input_shape), state[1].reshape(1,self._misc_len))
            a = np.argmax(qvals)      
        else:
            qvals = self._evaluate(state[0].reshape(self._single_image_input_shape))
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

    def _initialize_network(self, img_input_shape, misc_len, output_size, conv_layers=2, num_filters=[32, 32],
                            filter_size=[(5, 5), (5, 5)], hidden_units=[256], pool_size=[(2, 2), (2, 2)],
                            hidden_layers=1, conv_nonlin=rectify,
                            hidden_nonlin=leaky_rectify, output_nonlin=tanh):

        print "Initializing CNN ..."
        # image input layer
        network = ls.InputLayer(shape=img_input_shape, input_var=self._image_inputs)


        # convolution and pooling layers
        for i in range(conv_layers):
            network = ls.Conv2DLayer(network, num_filters=num_filters[i], filter_size=filter_size[i],
                                                 nonlinearity=conv_nonlin, W=relu_weights_initializer())
            network = ls.MaxPool2DLayer(network, pool_size=pool_size[i])
       
        network = ls.FlattenLayer(network)
        
        if self._misc_state_included:
            # misc input layer
            misc_input_layer = ls.InputLayer(shape=(None,misc_len), input_var=self._misc_inputs)
            # merge layer
            network = ls.ConcatLayer([network, misc_input_layer])

         # dense layers
        for i in range(hidden_layers):
            network = ls.DenseLayer(network, hidden_units[i], nonlinearity=hidden_nonlin, W=relu_weights_initializer())

        # output layer
        network = ls.DenseLayer(network, output_size, nonlinearity=output_nonlin)
        self._network = network

class LinearEvaluator(MLPEvaluator):
    def __init__(self, **kwargs):
        MLPEvaluator.__init__(self, **kwargs)

    def _initialize_network(self, img_input_shape, misc_len, output_size, output_nonlin = None):

        print "Initializing Linear evaluator ..."
        # image input layer
        network = ls.InputLayer(shape=img_input_shape, input_var=self._image_inputs)

        if self._misc_state_included:
            # misc input layer
            misc_input_layer = ls.InputLayer(shape=(None,misc_len), input_var=self._misc_inputs)
            # merge layer
            network = ls.ConcatLayer([network, misc_input_layer])


        # output layer
        network = ls.DenseLayer(network, output_size, nonlinearity=output_nonlin)
        self._network = network