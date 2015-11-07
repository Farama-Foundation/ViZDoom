
import numpy as np
from transition_bank import TransitionBank
import random

class QEngine:
	def __init__(self, game, evaluator, actions_generator, gamma = 0.7,batch_size = 500, update_frequency = 500, history_length = 1, bank_capacity = 10000, start_epsilon = 1.0, end_epsilon = 0.0,epsilon_decay_start_step = 100000, epsilon_decay_steps = 100000):
		self._game = game
		self._history_length = max( history_length,1)
		self._transitions = TransitionBank(bank_capacity)
		self._actions = actions_generator(game)
		self._actions_num = len(self._actions)
		self._actions_stats = np.zeros([self._actions_num],np.int) 
		self._update_frequency = update_frequency
		self._batch_size = batch_size
		self._epsilon = max(min(start_epsilon,1.0),0.0)
		self._end_epsilon = min(max(end_epsilon,0.0),self._epsilon)
		self._epsilon_decay_stride = (self._epsilon - end_epsilon)/epsilon_decay_steps
		self._epsilon_decay_start = epsilon_decay_start_step
		self._steps = 0
		self._gamma = gamma

		self.learning_mode = True

		#change img_shape according to the history size
		img_shape = game.get_state_format()[0]
		if history_length > 1:
			if len(img_shape)==2:
				img_shape = (self._history_length, img_shape[0],img_shape[1])
			else:
				img_shape = (self._history_length*img_shape[0], img_shape[1],img_shape[2])

		state_format = [img_shape,game.get_state_format()[1]]
		self._evaluator = evaluator(state_format, len(self._actions), batch_size)
		self._current_image_state = np.zeros(img_shape, dtype =  np.float32)

		if game.get_state_format()[1] > 0:
			self._misc_state_included = True

			self._current_misc_state = np.zeros(game.get_state_format()[1], dtype = np.float32)
		else:
			self._misc_state_included = False
	
	def _update_state(self):
		raw_state = self._game.get_state()

		if self._history_length > 1:
			self._current_image_state[0:-1] =  self._current_image_state[1:]
			self._current_image_state[-1] =  raw_state[0]
			if self._misc_state_included:
				self._current_misc_state[0:-1] = self._current_misc_state[1:]
				self._current_misc_state[-1] = np.array(raw_state[1:],dtype = np.float32)

		else:
			self._current_image_state =  raw_state[0]
			if self._misc_state_included:
				self._current_misc_state = np.array(raw_state[1:],dtype = np.float32)

	def _new_game(self):
		self._game.new_episode()
		self._current_image_state.fill(0.0)
		if self._misc_state_included:
			self._current_misc_state.fill(0.0)
		self._update_state()
	
	def make_step(self):
		if self.learning_mode:
			self._steps += 1
		#epsilon decay:
		if self._steps > self._epsilon_decay_start and self._epsilon >= 0:
			self._epsilon -= self._epsilon_decay_stride

		#if the current episode is finished, spawn a new one
		if self._game.is_finished():
			self._new_game()
		

		if self._misc_state_included:
			s = [self._current_image_state.copy(), self._current_misc_state.copy()]
		else:
			s = self._current_image_state.copy()
		
		if self.learning_mode:

			#with probability epsilon make random action:
			if self._epsilon >= random.random():
				a = random.randint(0,len(self._actions)-1)
			else:
				a = self._evaluator.best_action(s)
			
			self._actions_stats[a]+=1	
			_, r = self._game.make_action(self._actions[a])
			
			
			
			if self._game.is_finished():
				s2 = None
			else:
				self._update_state()
				if self._misc_state_included:
					s2 = [self._current_image_state.copy(), self._current_misc_state.copy()]
				else:
					s2 = self._current_image_state.copy()

			self._transitions.add_transition(s,a,s2,r)

			# Perform qlearning once for a while
			if self._steps % self._update_frequency == 0 and self._steps >self._batch_size:
				self._evaluator.learn(self._transitions.get_sample(self._batch_size), self._gamma)
		else:
			a = self._evaluator.best_action(s)
			self._actions_stats[a]+=1	
			self._game.make_action(self._actions[a])
			if not self._game.is_finished():
				self._update_state()

	def run_episode(self):
		self._new_game()
		while not self._game.is_finished():
			self.make_step()

		return self._game.get_summary_reward()


	def get_actions_stats(self,clear = False, norm = True):
		stats = self._actions_stats.copy()
		if norm:
			stats = stats/ np.float32(self._actions_stats.sum())
			stats [stats == 0.0] = -1
			stats = np.around(stats, 3)

		if clear:
			self._actions_stats.fill(0)
		return stats
