#!/usr/bin/python

import numpy as np
import agents
import games
import lasagne

def main():
	max_steps = np.inf
	test_n = 0
	print_stats = True
	print_qvalues = True
	
	mlp_agent_args={}
	
	mlp_agent_args['epsilon_decay_start_step'] = 500000
	mlp_agent_args['epsilon_decay_steps'] = 500000
	mlp_agent_args['start_epsilon'] = 1.0
	end_epsilon = 0.1
	mlp_agent_args['end_epsilon'] = min( end_epsilon, max(mlp_agent_args['start_epsilon'],end_epsilon))
	mlp_agent_args['gamma'] = 0.9
	network_params={}
	network_params['depth'] = 1
	network_params['hidden_units'] = 50
	network_params['input_dropout'] = False
	network_params['hidden_dropout'] = False
	network_params['input_dropout_p'] = 0.2
	network_params['hidden_dropout_p'] = 0.5
	network_params['loss_function'] = 'sgd'
	network_params['momentum'] = 0.9
	network_params['learning_rate'] = 0.01

	mlp_agent_args['network_params'] = network_params
	x=3
	y=1
	game = games.ShootingDotGame(width = x,height = y , max_moves = 50, miss_penalty = 0, living_reward = -1, hit_reward = 10, random_background = False)
	actions = len(game.actions)
	all_states = game.get_all_states()
	q_values = game.compute_qvalues(iterations = 30000, gamma = mlp_agent_args['gamma'])
	
	
	mlp_agent_args['game'] = game
	learner=agents.MLPQLearner(**mlp_agent_args)


	batch_size=100
	s_batch = np.ndarray([batch_size*x,1,y,x], dtype = np.float32)
	q_batch = np.ndarray([batch_size*x,actions], dtype = np.float32)
	
	
	for i in range(0,x*batch_size,x):
		
		s_batch[i:i+x] = all_states.copy()
		q_batch[i:i+x] = q_values.copy()
	
	print "Learning"
	n = 10000
	for i in range(n):
		q,loss=learner.Q_learn(s_batch,q_batch)
		#print loss

	learner.explore = False
	learner.learning_mode = False
	test_rewards = []
	norm_test_rewards = []
	for i in range(test_n):
		test_reward, norm_test_reward = learner.run_episode()
		test_rewards.append(test_reward)
		norm_test_rewards.append(norm_test_reward)

		test_mean = np.mean( test_rewards )
		norm_test_mean = np.mean( norm_test_rewards )
			
	if print_stats and test_n>0:
		print(str( round(norm_test_mean,4) )+ "\t" + str(test_mean))
		display_eps = round(learner.epsilon,2)
		print("stps:"+str(learner.steps)+", "+str(learner.actions_stats_test) + ", epsilon= "+ str(display_eps))
		

	if print_qvalues:
		print "LOSS:"
		print loss
		print "EXPECTED QVALUES:"
		print q_values
		print "LEARNED QVALUES:"
		print learner.learned_q_values()
	print "\nLast layer weights"
	print(lasagne.layers.get_all_param_values(learner.network)[-1])

main()