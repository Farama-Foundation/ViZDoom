import random


class TransitionBank:
    def __init__(self, capacity=10000):
        self._transitions = []
        self._size = 0
        self._capacity = capacity
        self._oldest_index = 0
    
    def add_transition(self, s1, a, s2, r):
        if self._size == self._capacity:
            self._transitions[self._oldest_index] = [s1, a, s2, r]
            self._oldest_index = (self._oldest_index + 1) % self._capacity

        else:
            self._transitions.append([s1, a, s2, r])
            self._size += 1

    def get_sample(self, batch_size):
        batch_size = min(batch_size, self._size)
        return random.sample(self._transitions, batch_size)
    
    def get_last_sample(self):
        return self._transitions[-1]
