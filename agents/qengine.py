import numpy as np
from transition_bank import TransitionBank
import random


class QEngine:
    def __init__(self, game, evaluator, actions_generator, gamma=0.7, batch_size=500, update_frequency=500,
                 history_length=1, bank_capacity=10000, start_epsilon=1.0, end_epsilon=0.0,
                 epsilon_decay_start_step=100000, epsilon_decay_steps=100000):
        self.online_mode = False
        self._game = game
        self._gamma = gamma
        self._batch_size = batch_size
        self._history_length = max(history_length, 1)
        self._update_frequency = update_frequency
        self._epsilon = max(min(start_epsilon, 1.0), 0.0)
        self._end_epsilon = min(max(end_epsilon, 0.0), self._epsilon)
        self._epsilon_decay_stride = (self._epsilon - end_epsilon) / epsilon_decay_steps
        self._epsilon_decay_start = epsilon_decay_start_step

        self.learning_mode = True
        self._transitions = TransitionBank(bank_capacity)
        self._steps = 0
        self._actions = actions_generator(game)
        self._actions_num = len(self._actions)
        self._actions_stats = np.zeros([self._actions_num], np.int)

        
        # change img_shape according to the history size
        self._single_img_shape = list(game.get_state_format()[0])
        if len(self._single_img_shape) == 2:
            self._single_img_shape = [1,self._single_img_shape[0],self._single_img_shape[1]]
        self._channels = self._single_img_shape[0]

        img_shape = self._single_img_shape
        if history_length > 1:
            img_shape[0] *= history_length

        state_format = [img_shape, game.get_state_format()[1]]
        self._evaluator = evaluator(state_format, len(self._actions), batch_size, self._gamma)
        self._current_image_state = np.zeros(img_shape, dtype=np.float32)

        if game.get_state_format()[1] > 0:
            self._misc_state_included = True
            self._current_misc_state = np.zeros(game.get_state_format()[1], dtype=np.float32)
        else:
            self._misc_state_included = False

    def _update_state(self):
        raw_state = self._game.get_state()
        img = raw_state[0].copy()

        if self._misc_state_included:
            misc = raw_state[1].copy()
        if self._history_length > 1:
            self._current_image_state[0:-self._channels] = self._current_image_state[self._channels:]
            self._current_image_state[-self._channels:] = raw_state[0].copy()
            if self._misc_state_included:
                self._current_misc_state[0:-1] = self._current_misc_state[1:]
                self._current_misc_state[-1] = misc

        else:
            self._current_image_state[:] = img
            if self._misc_state_included:
                self._current_misc_state = misc

    def _new_game(self):
        self._game.new_episode()
        self._current_image_state.fill(0.0)
        if self._misc_state_included:
            self._current_misc_state.fill(0.0)
        self._update_state()

    def make_step(self):
        if self.learning_mode:
            self._steps += 1
        # epsilon decay:
        if self._steps > self._epsilon_decay_start and self._epsilon > 0:
            self._epsilon -= self._epsilon_decay_stride
            self._epsilon = max(self._epsilon, 0)

        # if the current episode is finished, spawn a new one
        if self._game.is_finished():
            self._new_game()

        if self._misc_state_included:
            s = [self._current_image_state.copy(), self._current_misc_state.copy()]
        else:
            s = [self._current_image_state.copy()]

        if self.learning_mode:

            # with probability epsilon make random action:
            if self._epsilon >= random.random():
                a = random.randint(0, len(self._actions) - 1)
            else:
                a = self._evaluator.best_action(s)

            self._actions_stats[a] += 1
            r = self._game.make_action(self._actions[a])

            if self._game.is_finished():
                s2 = None
            else:
                self._update_state()
                if self._misc_state_included:
                    s2 = [self._current_image_state.copy(), self._current_misc_state.copy()]
                else:
                    s2 = [self._current_image_state.copy()]

            self._transitions.add_transition(s, a, s2, r)

            # Perform q-learning once for a while
            if self._steps % self._update_frequency[0] == 0 and not self.online_mode and self._steps > self._batch_size:
                for i in range(self._update_frequency[1]):
                    self._evaluator.learn(self._transitions.get_sample(self._batch_size))
            elif self.online_mode:
                self._evaluator.learn_one(s, a, s2, r)
        else:
            a = self._evaluator.best_action(s)
            self._actions_stats[a] += 1
            self._game.make_action(self._actions[a])
            if not self._game.is_finished():
                self._update_state()

    def run_episode(self):
        self._new_game()
        while not self._game.is_finished():
            self.make_step()

        return self._game.get_summary_reward()

    def get_actions_stats(self, clear=False, norm=True):
        stats = self._actions_stats.copy()
        if norm:
            stats = stats / np.float32(self._actions_stats.sum())
            stats[stats == 0.0] = -1
            stats = np.around(stats, 3)

        if clear:
            self._actions_stats.fill(0)
        return stats

    def get_steps(self):
        return self._steps

    def get_epsilon(self):
        return self._epsilon
