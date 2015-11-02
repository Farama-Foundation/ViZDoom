import numpy as np
import random
import theano
import theano.tensor as T
import lasagne


class RandomAgent:

	def __init__(self, game):
		#Whether learning should happen in makeAction
		self.learning_mode = True
		#Whether exploration should be done according to epsilon
		self.explore = True
		self.game = game
		self.actions = self.game.get_action_format()[0]["range"][1]+1
		self.steps = 0
		self.actions_stats_test = np.zeros(self.actions)
		self.actions_stats_learning = np.zeros(self.actions)

	def clear_actions_stats(self):
		self.actions_stats_test.fill(0)
		self.actions_stats_learning.fill(0)

	def make_action(self):
		move = random.randint(0,self.actions-1)
		self.actions_stats_test[move] += 1
		self.game.make_action([move])
		self.steps += 1

	def run_episode(self):
		self.game.new_episode()
		while not self.game.is_finished():
			self.make_action()
		reward = self.game.get_summary_reward()
		normalized_reward = self.game.get_normalized_summary_reward()
			

		return reward, normalized_reward

class HumanAgent(RandomAgent):
	def __init__(self,game):
		RandomAgent.__init__(self, game)
		self.current_score = 0
	def make_action(self):
		print self.game.get_state()
		move=raw_input()

		if move =='a':
			move = 0
		elif move =='d':
			move = 1
		elif move =='s':
			move = 2
		else:
			move = 3

		state,reward = self.game.make_action([move])
		self.current_score += reward
		print("Current summary reward: " + str(self.current_score))


	def run_episode(self):
		print("\nNew episode")
		r, norm_r = RandomAgent.run_episode(self)
		print("\nEpisode finished")
		self.current_score = 0

		return r, norm_r

class MLPQLearner(RandomAgent):

	def __init__(self,game,gamma = 0.99, epsilon_decay_start_step = 0, start_epsilon = 1.0,end_epsilon = 0.1,epsilon_decay_steps=100000, 
		network_params={'depth':2, 'hidden_units':100,'input_dropout': True,'hidden_dropout': True,
		'input_dropout_p': 0.2,'hidden_dropout_p': 0.5}):

		RandomAgent.__init__(self,game)
		
		self.network_params = network_params
		self.epsilon = start_epsilon
		self.start_epsilon = start_epsilon 
		self.end_epsilon = end_epsilon
		self.epsilon = start_epsilon
		self.epsilon_decay_steps = epsilon_decay_steps
		self.epsilon_decay_stride = (start_epsilon - end_epsilon)/epsilon_decay_steps
		self.initialize_network()
		self.gamma = gamma
		self.epsilon_decay_start_step = epsilon_decay_start_step

	def make_action(self):
		if self.learning_mode:
			self.steps += 1

			s = self.game.get_state()
			s = s.reshape(1,1,s.shape[0],s.shape[1])
			predicted_Qs = self.Q_test(s)
			predicted_Qs_buffered = predicted_Qs.copy()

			if self.explore and random.random() <= self.epsilon:
				#make a random move
				a = random.randint(0,self.actions-1)
			else:	
				#make the best move				
				a = np.argmax(predicted_Qs)
			
			
			s2, r = self.game.make_action([a])

			self.actions_stats_learning[a] += 1
			expected_Q = r

			if not self.game.is_finished():
				s2 = s2.reshape(1,1,s2.shape[0],s2.shape[1])
				best_q2 = max(self.Q_test(s2)[0])
				expected_Q += self.gamma * best_q2


			predicted_Qs[0][a]=expected_Q

			q,l = self.Q_learn(s,predicted_Qs)


			if self.epsilon_decay_start_step <= self.steps:
				self.epsilon =max(self.epsilon- self.epsilon_decay_stride,self.end_epsilon)
		else:
			s = self.game.get_state()
			s = s.reshape(1,1,s.shape[0], s.shape[1])
			predicted_Qs = self.Q_test(s)
			a = np.argmax(predicted_Qs)
			self.game.make_action([a])
			self.actions_stats_test[a] += 1

		#make not-so-random move which is not supported yet



	def initialize_network(self):
		print("Initializing MLP network")
		
		
		hidden_layers = self.network_params["depth"]
		hidden_units = self.network_params["hidden_units"]
		input_dropout = self.network_params["input_dropout"]
		hidden_dropout = self.network_params["hidden_dropout"]
		learning_rate = self.network_params["learning_rate"]
		momentum = self.network_params["momentum"]
		
		# input layer
		inputs = T.tensor2('inputs')
		targets = T.matrix('targets')
		y,x = self.game.get_state_format()[0]["shape"]
		network = lasagne.layers.InputLayer(shape=(None,1,y,x),input_var = inputs)
		# input dropout layer
		if input_dropout: 
			input_dropout_p = self.network_params["input_dropout_p"]
			if input_dropout_p > 0:
				network = lasagne.layers.dropout(network, p = input_dropout_p)

		nonlin = lasagne.nonlinearities.tanh

		# hidden units with dropouts
		for layer_i in range(hidden_layers):
			network = lasagne.layers.DenseLayer(network, hidden_units, nonlinearity = nonlin)
			if hidden_dropout:
				hidden_dropout_p = self.network_params["hidden_dropout_p"]
				if hidden_dropout_p > 0:
					network = lasagne.layers.dropout(network, p = hidden_dropout_p)

		# output layer
		network = lasagne.layers.DenseLayer(network, self.actions, nonlinearity = None)
		self._network = network
		


		#Theano stuff
		q_prediction = lasagne.layers.get_output(network)
		loss = lasagne.objectives.squared_error(q_prediction, targets).mean()
		params = lasagne.layers.get_all_params(network, trainable = True)
		
		if self.network_params["loss_function"] == 'nesterov_momentum':
			updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate = learning_rate, momentum=momentum)
		elif self.network_params["loss_function"] == 'sgd':
			updates = lasagne.updates.sgd(loss, params, learning_rate = learning_rate)
		else:
			print ("Unsupported loss function: "+str(self.network_params["loss_function"]))
			exit(1)

		test_q_prediction = lasagne.layers.get_output(network, deterministic = True)

		print "\tCompiling network functions."
		self.Q_learn = theano.function([inputs, targets], [q_prediction, loss], updates = updates)
		self.Q_test = theano.function([inputs], test_q_prediction)
		print "\tNetwork functions compiled."

	def nan_check(self):
		params = lasagne.layers.get_all_param_values(self._network)
		for array in params:
			if np.isnan(array).any():
				return True
		return False