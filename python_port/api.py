import numpy as np


class Api:

	def __init__(self):
		self._finished = True
		self._state = None
		self._summary_reward = None
		#sample _action_format
		self._action_format = []
		self._action_format.append({"name":"mouse_horizontal_delta","dtype":np.float32,"range":[-1,1]})
		self._action_format.append({"name":"a_key","dtype":np.bool_})
		self._action_format.append({"name":"d_key","dtype":np.bool_})
		self._action_format.append({"name":"w_key","dtype":np.bool_})
		self._action_format.append({"name":"ctrl","dtype":np.bool_})
		#sample action: [0.0,False,False,False,False]
		#sample _state_format
		self._state_format = [{"name":"image","dtype":np.float32,"range":[0.0,1.0],"shape":(3,320,200)},{"name":"hp","dtype":np.float32,"range":[0,100]}]
		
	def get_state_format(self):
		return self._state_format
	
	def get_action_format(self):
		return self._action_format

	def make_action(self, action):
		if self._finished:
			#throw some exception?
			return
		reward = 0
		self._summary_reward += reward
		return reward, self._state

	def get_summary_reward(self):
		return self._summary_reward

	def is_finished(self):
		return self._finished

	def new_episode(self):
		self._finished = False
		self._summary_reward = 0
		return self._state
