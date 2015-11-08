import random
import lasagne
import theano.tensor as T
from theano.tensor import tanh
import theano
import numpy as np
from theano.compile.nanguardmode import NanGuardMode

class MLPEvaluator:

	def __init__(self, state_format, actions_number, batch_size, network_args):
		print "Initializing MLP network..."
		self._misc_state_included = (state_format[1] > 0)
		self.online_mode = False

		if self._misc_state_included:
			self._misc_inputs = T.matrix('misc_inputs')
			misc_input_shape = (None, state_format[1])
			self._misc_input_shape = (1, state_format[1])
			self._misc_buffer = np.ndarray((batch_size, state_format[1]),dtype = np.float32)
			self._misc_buffer2 = np.ndarray((batch_size, state_format[1]),dtype = np.float32)
		else:
			misc_input_shape = None
		image_dimensions = len(state_format[0])

		self._targets = T.matrix('targets')
		if image_dimensions == 2:
			self._image_inputs = T.tensor3('image_inputs')
		elif image_dimensions == 3:
			self._image_inputs = T.tensor4('image_inputs')

		image_input_shape = list(state_format[0])
		image_input_shape.insert(0,None)
		self._image_input_shape = list(image_input_shape)
		self._image_input_shape[0] = batch_size

		#create buffers for batch learning
		self._input_image_buffer = np.ndarray(self._image_input_shape,dtype = np.float32)
		self._input_image_buffer2 = np.ndarray(self._image_input_shape,dtype = np.float32)
		self._expected_buffer = np.ndarray([batch_size], dtype = np.float32)

		#save it for the evaluation reshape
		self._image_input_shape[0] = 1

		network_args["img_shape"] = image_input_shape
		network_args["misc_shape"] = misc_input_shape 
		network_args["output_size"] = actions_number
		self._initialize_network(**network_args)
		
	def _initialize_network(self,img_shape, misc_shape,output_size, hidden_units = [500], learning_rate =0.01,hidden_layers = 1, hidden_nonlin = lasagne.nonlinearities.tanh, updates = lasagne.updates.sgd):
		#image input layer
		network = lasagne.layers.InputLayer(shape = img_shape, input_var = self._image_inputs)
		#hidden layers
		for i in range(hidden_layers):
			network = lasagne.layers.DenseLayer(network, hidden_units[i] ,nonlinearity = hidden_nonlin)	
		if self._misc_state_included:
			#misc input layer
			misc_input_layer = lasagne.layers.InputLayer(shape = misc_shape, input_var = self._misc_inputs)
			#merge layer
			network = lasagne.layers.ConcatLayer([network, misc_input_layer])
			
		#output layer
		network = lasagne.layers.DenseLayer(network, output_size, nonlinearity = None)
		self._network = network
		
		predictions = lasagne.layers.get_output(network)
		loss =  lasagne.objectives.squared_error(predictions, self._targets).mean()
		params = lasagne.layers.get_all_params(network, trainable = True)
		updates = updates(loss, params, learning_rate = learning_rate)
		
		#mode = NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
		mode = None
		if self._misc_state_included:
			self._learn = theano.function([self._image_inputs,self._misc_inputs,self._targets], loss, updates = updates, mode = mode, name = "learn_fn")
			self._evaluate = theano.function([self._image_inputs,self._misc_inputs], predictions,mode = mode, name = "eval_fn")
		else:	
			self._learn = theano.function([self._image_inputs,self._targets], loss, updates = updates)
			self._evaluate = theano.function([self._image_inputs], predictions)
	
	def learn(self, transitions, gamma):
		 
		#TODO:
		#change internal representation of transitions so that it would return
		#ready ndarrays
		#prepare the batch
		
		if self._misc_state_included:
			for i,trans in zip(range(len(transitions)),transitions):
				self._input_image_buffer[i] = trans[0][0]
				self._misc_buffer[i] = trans[0][1]
				# if it's the terminal state just ignore
				if trans[2] is not None:		
					self._input_image_buffer2[i] = trans[2][0]
					self._misc_buffer2[i] = trans[2][1]
			
			target = self._evaluate(self._input_image_buffer,self._misc_buffer)
			#find best q values for s2
			q2 = np.max(self._evaluate(self._input_image_buffer2,self._misc_buffer2),axis = 1)

		else:
			for i,trans in zip(range(len(transitions)),transitions):
				
				self._input_image_buffer[i] = trans[0]
				# if it's the terminal state just ignore
				if trans[2] is not None:		
					self._input_image_buffer2[i] = trans[2]

			target = self._evaluate(self._input_image_buffer)
			#find best q values for s2
			q2 = np.max(self._evaluate(self._input_image_buffer2),axis = 1)
		
		#set expected output as the reward got from the transition
		for i,trans in zip(range(len(transitions)),transitions):
			self._expected_buffer[i] = trans[3]

		#substitute expected values for chosen actions 
		for i,q in zip(range(len(transitions)),q2):
			if transitions[i][2] is not None:
				self._expected_buffer[i] += gamma *q
			target[i][transitions[i][1]] =self._expected_buffer[i]
		
		if self._misc_state_included:
			self._learn(self._input_image_buffer,self._misc_buffer, target)
		else:
			self._learn(self._input_image_buffer,target)
		

	def best_action(self, state):
		if self._misc_state_included:
			a = np.argmax(self._evaluate(state[0].reshape(self._image_input_shape),state[1].reshape(self._misc_input_shape)))
		else:
			a = np.argmax(self._evaluate(state.reshape(self._image_input_shape)))
		return a





