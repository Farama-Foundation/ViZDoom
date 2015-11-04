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
		self._state_format = [(self._y, self._x),0]
		self._action_format = [{"name":"action","dtype":np.int8,"range":[0,3]}]
		self._state = None
		self._finished = True
		self._summary_reward = 0

	def is_finished(self):
		return self._finished

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
			print "Making action in a finished game."
			return None
		else:
			action = action[0]
			reward=self._living_reward
			self._movesMade += 1

			#right
			if action == 0:
				if self._aimX>0:
					self._state[self._aimY,self._aimX] = 0.0
					self._aimX -= 1
					self._state[self._aimY,self._aimX] = 1.0
			#left
			elif action == 1:
				if self._aimX < self._state.shape[1]-1:
					self._state[self._aimY,self._aimX] = 0.0
					self._aimX += 1
					self._state[self._aimY,self._aimX] = 1.0
			#shoot
			elif action == 2:
				if self._aimX != self._state.shape[1]/2:
					reward -= self._miss_penalty
				else:
					reward += self._hit_reward
					self._finished = True;
					self._state = None
			elif action != 3:
				print "Unknown action. Idle action chosen."
			
			self._summary_reward+=reward

			if self._movesMade >= self._max_moves:
				self._finished = True
				self._state = None

			if self._state is None:
				return None, reward
			else:
				return [self._state.copy()], reward

	def get_state(self):
		if self._state is None:
			return None
		else:
			return [self._state.copy()]
