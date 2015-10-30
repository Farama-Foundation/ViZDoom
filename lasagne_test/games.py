import numpy as np
import random 
import sys

class ShootingDotGame:
	
	def __init__(self, width, height, dtype = np.float32, random_background = False,max_moves=np.inf,living_reward=-1,miss_penalty=10,hit_reward=100):
		
		self._dtype = dtype
		width+=1-width%2
		self._x = width
		self._y = height
		self._movesMade = 0
		self._living_reward = living_reward
		self._miss_penalty = miss_penalty
		self._hit_reward = hit_reward
		self._max_moves = max_moves
		self._random_background = random_background
		self._state_format = [{"name":"image","dtype":np.float32,"range":[0.0,1.0],"shape":(self._y, self._x)}]
		self._action_format = None
		self._state = None
		self._finished = True
		self._summary_reward = 0

	def is_finished(self):
		return self._is_finished

	def get_state_format(self):
		return self._state_format
	
	def get_action_format(self):
		return self._action_format

	def get_normalized_summary_reward(self):
		return self._summary_reward / self._max_reward

	def get_summary_reward(self):
		return self._summary_reward

	def new_episode(self):
		self._finished = False
		self._movesMade = 0
		self._aimX = random.randint(0,self._x-1)
		self._aimY = int(self._y/2)
		if self._random_background:
			self._state = self._dtype(np.random.rand(self._y,self._x))
			self._state[self._aimY] = 0.0
		else:
			self._state = np.zeros([self._y,self._x],dtype = self._dtype)
		
		self._state[self._aimY,self._aimX] = 1.0
		self._summary_reward = 0
		self._max_reward = float(self._living_reward * (abs(self._aimX - self._state.shape[1]/2) + 1) +self._hit_reward)
		
		if self._max_moves == np.inf:
			self._min_reward = -np.inf
		else:
			self._min_reward = float(self._max_moves*(min(0.0,-self._miss_penalty)+self._living_reward))

	def make_action(self,action):
		if self._finished:
			return None
		else:
			reward=self._living_reward
			self._movesMade += 1

			if self._movesMade >= self._max_moves:
				self._finished = True
			#right
			if action == 0:
				if self._aimX>0:
					self._state[self._aimY,self._aimX] = 0.0
					self._aimX -= 1
					self._state[self._aimY,self._aimX] = 1.0
			#left
			elif action == 1:
				if self._aimX<self._state.shape[1]-1:
					self._state[self._aimY,self._aimX] = 0.0
					self._aimX += 1
					self._state[self._aimY,self._aimX] = 1.0
			#shoot
			elif action == 2:
				if self._aimX != self._state.shape[1]/2:
					reward -= self._miss_penalty
				else:
					reward += self._hit_reward
					self._state = None
					self._finished = True;
			elif action != 3:
				print "Unknown action. Idle action chosen."
			self._summary_reward+=reward
			return self._state, reward
			
	def compute_qvalues(self,iterations=50000, learning_rate =0.1, gamma = 1.0):
		state_transformator = np.asarray(range(self._x))

		Q = np.zeros([self._x, self._actions_num],dtype = self._dtype)
		self._reset()
		for i in range(iterations):
			if self._finished:
				self._reset()
			a = random.randint(0,self._actions_num-1)
			s = np.dot(self._state, state_transformator)[0]

			s2,r = self._make_action(a)
			best_q2 = 0
			if not self._finished:
				s2 = np.dot(s2,state_transformator)[0]
				best_q2 = Q[s2].max()
			Q[s,a] += learning_rate *(r +gamma*best_q2-Q[s,a])
		self._reset()
		return np.around(Q,2)