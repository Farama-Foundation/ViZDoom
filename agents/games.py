import numpy as np
import random


class ShootingDotGame:
    def __init__(self, x, y, add_dimension=False, dtype=np.float32, random_background=False, max_moves=np.inf,
                 living_reward=-1, miss_penalty=10, hit_reward=100, ammo=np.inf):
        self._ammo = np.float32(max(ammo, 0))
        self._dtype = dtype
        x += 1 - x % 2
        self._x = x
        self._y = y
        self._moves_made = 0
        self._living_reward = living_reward
        self._miss_penalty = miss_penalty
        self._hit_reward = hit_reward
        self._max_moves = max_moves
        self._random_background = random_background
        self._add_dimension = add_dimension
        if self._add_dimension:
            self._state_format = [(1, self._y, self._x)]
        else:
            self._state_format = [(self._y, self._x)]

        if ammo < np.inf:
            self._state_format.append(1)
        else:
            self._state_format.append(0)
        self._state_format = tuple(self._state_format)
        self._current_ammo = np.ndarray([1], dtype=np.float32)
        self._action_format = 3
        self._state = None
        self._finished = True
        self._summary_reward = 0
        self._aimX = -1
        self._aimY = -1

    def is_finished(self):
        return self._finished

    def get_state_format(self):
        return self._state_format

    def get_action_format(self):
        return self._action_format

    def get_normalized_summary_reward(self):
        return self._summary_reward / self._hit_reward

    def get_summary_reward(self):
        return self._summary_reward

    def new_episode(self):
        self._current_ammo[0] = self._ammo
        self._finished = False
        self._moves_made = 0
        self._aimX = random.randint(0, self._x - 1)
        self._aimY = int(self._y / 2)
        if self._random_background:
            self._state = self._dtype(np.random.rand(self._y, self._x))
            self._state[self._aimY] = 0.0
        else:
            self._state = np.zeros([self._y, self._x], dtype=self._dtype)

        self._state[self._aimY, self._aimX] = 1.0
        self._summary_reward = 0

    def make_action(self, action):

        if self._finished:
            print "Making action in a finished game."
            return None
        else:

            reward = self._living_reward
            self._moves_made += 1

            # right
            if action[0] and not action[1]:
                if self._aimX > 0:
                    self._state[self._aimY, self._aimX] = 0.0
                    self._aimX -= 1
                    self._state[self._aimY, self._aimX] = 1.0
            # left
            elif action[1] and not action[0]:
                if self._aimX < self._state.shape[1] - 1:
                    self._state[self._aimY, self._aimX] = 0.0
                    self._aimX += 1
                    self._state[self._aimY, self._aimX] = 1.0
            # shoot
            if action[2]:
                if self._current_ammo[0] > 0:
                    if self._aimX != self._state.shape[1] / 2:
                        reward -= self._miss_penalty
                    else:
                        reward += self._hit_reward
                        self._finished = True
                        self._state = None
                    self._current_ammo[0] -= 1

            self._summary_reward += reward

            if self._moves_made >= self._max_moves:
                self._finished = True
                self._state = None

            return reward

    def get_state(self):
        if self._state is None:
            return None
        else:
            ret_val = [self._state.reshape(self._state_format[0])]
            if self._ammo < np.inf:
                ret_val.append(self._current_ammo / self._ammo)

            return ret_val

    def average_best_result(self):
        best = self._hit_reward + self._living_reward
        worst = self._hit_reward + self._living_reward * (self._x - 1) / 2.0
        avg = (best + worst) / 2.0
        r = (best + (self._x - 1) * avg) / float(self._x)
        return r
