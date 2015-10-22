import numpy as np
import random 
import sys

class ShootingDotGame:
	actions_labels = ("left","right","shoot","idle")
	actions = (0,1,2,3)	

	def __init__(self, width, height, random_background = False,max_moves=np.inf,living_reward=-1,miss_penalty=10,hit_reward=100):
		
		width+=1-width%2
		self.x = width
		self.y = height
		self.movesMade = 0
		self.living_reward = living_reward
		self.miss_penalty = miss_penalty
		self.hit_reward = hit_reward
		self.max_moves = max_moves
		self.random_background = random_background
		self.reset()

	def get_normalized_summary_reward(self):
		
		return self.summary_reward / self.max_reward

	def reset(self):
		self.finished = False
		self.movesMade = 0
		self.aimX = random.randint(0,self.x-1)
		self.aimY = int(self.y/2)
		if self.random_background:
			self.state = np.float32(np.random.rand(self.y,self.x))
			self.state[self.aimY] = 0.0
		else:
			self.state = np.zeros([self.y,self.x],dtype=np.float32)
		
		self.state[self.aimY,self.aimX] = 1.0
		self.summary_reward = 0
		self.max_reward = float(self.living_reward * (abs(self.aimX - self.state.shape[1]/2) + 1) +self.hit_reward)
		if self.max_moves == np.inf:
			self.min_reward = -np.inf
		else:
			self.min_reward = float(self.max_moves*(min(0.0,-self.miss_penalty)+self.living_reward))

	def make_action(self,action):
		if self.finished:
			return None
		else:
			reward=self.living_reward
			self.movesMade += 1

			if self.movesMade >= self.max_moves:
				self.finished = True
			#right
			if action == 0:
				if self.aimX>0:
					self.state[self.aimY,self.aimX] = 0.0
					self.aimX -= 1
					self.state[self.aimY,self.aimX] = 1.0
			#left
			elif action == 1:
				if self.aimX<self.state.shape[1]-1:
					self.state[self.aimY,self.aimX] = 0.0
					self.aimX += 1
					self.state[self.aimY,self.aimX] = 1.0
			#shoot
			elif action == 2:
				if self.aimX != self.state.shape[1]/2:
					reward -= self.miss_penalty
				else:
					reward += self.hit_reward
					self.state = None
					self.finished = True;
			elif action != 3:
				print "Unknown action. Idle action chosen."
			self.summary_reward+=reward
			return self.state, reward
			




