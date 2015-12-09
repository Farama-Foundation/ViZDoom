import random


class TransitionBank:
    def __init__(self, capacity=10000, rejection_range = None, rejection_probability=0.0):
        self._transitions = []
        self._size = 0
        self._capacity = capacity
        self._oldest_index = 0
        self._rejection_range = rejection_range
        self._rejection_probability = rejection_probability
    
    def add_transition(self, s1, a, s2, r):
        if self._size == self._capacity:
            #ignore the transition if it's not very
            if self._rejection_range and self._rejection_probability>0:
                if r >self._rejection_range[0] and r<self._rejection_range[1]:
                    if random.random() > self._rejection_probability:
                        return
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
