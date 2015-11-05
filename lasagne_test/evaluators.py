import random
import lasagne
import theano.tensor as T
from theano.tensor import tanh
import theano
import numpy as np


class MLPEvaluator:

	def __init__(self, state_format, actions_number, batch_size):
		print "Initializing MLP network..."
		self._misc_state_included = (state_format[1] > 0)
		
		image_dimensions = len(state_format[0])

		targets = T.matrix('targets')
		if image_dimensions == 2:
			image_inputs = T.tensor3('image_inputs')
		elif image_dimensions == 3:
			image_inputs = T.tensor4('image_inputs')

		image_input_shape = list(state_format[0])
		image_input_shape.insert(0,None)
		self._image_input_shape = list(image_input_shape)
		self._image_input_shape[0] = batch_size

		#create buffers for batch learning
		self._input_image_buffer = np.ndarray(self._image_input_shape,dtype = np.float32)
		self._input_image_buffer2 = np.ndarray(self._image_input_shape,dtype = np.float32)
		self._expected_buffer = np.ndarray([batch_size], dtype = np.float32)

		#remember for the evaluation reshape
		self._image_input_shape[0] = 1


		learning_rate = 0.01
		hidden_units = 1000
	

		network = lasagne.layers.InputLayer(shape = image_input_shape, input_var = image_inputs)
		network = lasagne.layers.DenseLayer(network, hidden_units ,nonlinearity = tanh)
		network = lasagne.layers.DenseLayer(network, actions_number, nonlinearity = None)
		self._network = network
		
		predictions = lasagne.layers.get_output(network)
		loss =  lasagne.objectives.squared_error(predictions, targets).mean()
		params = lasagne.layers.get_all_params(network, trainable = True)
		updates = lasagne.updates.sgd(loss, params, learning_rate = learning_rate)
		
		self._learn = theano.function([image_inputs,targets], loss, updates = updates)
		self._evaluate = theano.function([image_inputs], predictions)
		

	def learn(self, transitions, gamma):
		 
		#TODO:
		#change internal representation of transitions so that it would return
		#ready ndarrays
		#prepare the batch
		
		for i,trans in zip(range(len(transitions)),transitions):
			#trans[0] is the whole state in the transition
			#trans[0][0] - image
			#trans[0][1] - misc data from the state
			self._input_image_buffer[i] = trans[0][0]
			# if it's the terminal state just ignore
			if trans[2] is not None:		
				self._input_image_buffer2[i] = trans[2][0]

		target = self._evaluate(self._input_image_buffer)
		#best q values for s2
		q2 = np.max(self._evaluate(self._input_image_buffer2),axis = 1)
		
		#set expected output as the reward got from the transition
		for i,trans in zip(range(len(transitions)),transitions):
			self._expected_buffer[i] = trans[3]

		#substitute expected values for chosen actions 
		for i,q in zip(range(len(transitions)),q2):
			if transitions[i][2] is not None:
				self._expected_buffer[i] += gamma *q
			target[i][transitions[i][1]] =self._expected_buffer[i]
		

		self._learn(self._input_image_buffer,target)

	def best_action(self, state):
		a = np.argmax(self._evaluate(state[0].reshape(self._image_input_shape)))
		return a





