
import numpy as np
from transition_bank import TransitionBank

class QEngine:
	def __init__(self, game, evaluator, actions_generator, history_length = 1, bank_capacity = 10000, start_epsilon = 1.0, end_epsilon = 0.0,epsilon_decay_start_step = 100000, epsilon_decay_steps = 100000):
		self._game = game
		self._history_length = history_length
		self._transitions = TransitionBank(bank_capacity)
		self._actions = actions_generator(game.get_action_format())
		self._actions_num =len(self._actions)
		self._evaluator = evaluator(game.get_state_format(), self._actions)

		img_shape = game.get_state_format()[0]["shape"]
		if len(img_shape)==2:
			img_shape = (self._history_length, img_shape[0],img_shape[1])
		else:
			img_shape = (self._history_length*img_shape[0], img_shape[1],img_shape[2])

		self._current_image_state = np.zeros(img_shape, dtype =  np.float32)

		if len(game.get_state_format())>1:
			self._misc_state_included = True
			self._current_misc_state = np.zeros(len(game.get_state_format())-1, dtype = np.float32)
		else:
			self._misc_state_included = False
		