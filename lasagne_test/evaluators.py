import random

class CNNEvaluator:

	def __init__(self, state_format, actions_number):
		self._misc_state_included = (len(state_format) >1)
		
		self.n = actions_number - 2
		None

	def learn(self, transitions):
		None
	
	def evaluate(self, state):
		return random.randint(0,self.n)
		return self._evaluate(state)

	def _initialize_network(self):
		self.evaluate = None
		self._learn = None
		None
