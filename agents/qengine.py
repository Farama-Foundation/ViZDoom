import numpy as np
from transition_bank import TransitionBank
import random


class QEngine:
    def __init__(self, game, evaluator, actions_generator, gamma=0.7, batch_size=500, update_frequency=500,
                 history_length=1, bank = None, bank_capacity=10000, start_epsilon=1.0, end_epsilon=0.0,
                 epsilon_decay_start_step=100000, epsilon_decay_steps=100000, reward_scale = 1.0):
        self.online_mode = False
        self._reward_scale = reward_scale
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
        if bank and type(bank) == type(TransitionBank):
            self._transitions = bank
        else:
            self._transitions = TransitionBank(bank_capacity)
        self._steps = 0
        self._actions = actions_generator(game)
        self._actions_num = len(self._actions)
        self._actions_stats = np.zeros([self._actions_num], np.int)

        
        # change img_shape according to the history size
        self._channels = game.get_screen_channels()
        img_shape = [self._channels, game.get_screen_width(), game.get_screen_height()]
        
        if history_length > 1:
            img_shape[0] *= history_length
        
        state_format = [img_shape, game.get_game_var_len()]
        self._evaluator = evaluator(state_format, len(self._actions), batch_size, self._gamma)
        self._current_image_state = np.zeros(img_shape, dtype=np.float32)

        if game.get_game_var_len() > 0:
            self._misc_state_included = True
            self._current_misc_state = np.zeros(game.get_game_var_len(), dtype=np.float32)
        else:
            self._misc_state_included = False

    # it doesn't copy the state
    def _update_state(self, raw_state):
        img = np.float32(raw_state[1])/255.0
        if self._misc_state_included:
            misc = raw_state[2]
        if self._history_length > 1:
            self._current_image_state[0:-self._channels] = self._current_image_state[self._channels:]
            self._current_image_state[-self._channels:] = img
            if self._misc_state_included:
                self._current_misc_state[0:-1] = self._current_misc_state[1:]
                self._current_misc_state[-1] = misc

        else:
            self._current_image_state[:] = img
            if self._misc_state_included:
                self._current_misc_state = misc
        
    def _new_game(self):
        self._game.new_episode()
        self.reset_state()
        raw_state = self._game.get_state()
        self._update_state(raw_state)        

    def _copy_current_state(self):
    	if self._misc_state_included:
            s = [self._current_image_state.copy(), self._current_misc_state.copy()]
        else:
            s = [self._current_image_state.copy()]
        return s
    
    def _current_state(self):
    	if self._misc_state_included:
            s = [self._current_image_state, self._current_misc_state]
        else:
            s = [self._current_image_state]
        return s

    def _choose_action_index(self, state, verbose = False):
    	s = self._current_state()
    	return self._evaluator.best_action(s, verbose = verbose)

    def learn_from_master(self, state_action):
    	pass
        #TODO

    def choose_action(self, state, verbose = False):
        self._update_state(state)
        return self._actions(self._choose_action_index(state, verbose))

    def reset_state(self):
        self._current_image_state.fill(0.0)
        if self._misc_state_included:
            self._current_misc_state.fill(0.0)

    def make_step(self, verbose = False):
    	raw_state = self._game.get_state()
        a = self._choose_action_index(raw_state, verbose)
    	self._actions_stats[a] += 1
    	self._game.make_action(self._actions[a])
        if not self._game.is_episode_finished():
            raw_state = self._game.get_state()
            self._update_state(raw_state)

    def make_learning_step(self, verbose = False):
        self._steps += 1
       	# epsilon decay:
        if self._steps > self._epsilon_decay_start and self._epsilon > 0:
	        self._epsilon -= self._epsilon_decay_stride
	        self._epsilon = max(self._epsilon, 0)

	    # copy needed as it will be stored in transitions
        s = self._copy_current_state();

        # with probability epsilon choose a random action:
        if self._epsilon >= random.random():
            a = random.randint(0, len(self._actions) - 1)
        else:
            a = self._evaluator.best_action(s, verbose = verbose)
        self._actions_stats[a] += 1

        # make action and get the reward
        r = self._game.make_action(self._actions[a])
        r = np.float32(r)*self._reward_scale
        #update state s2 accordingly
        if self._game.is_episode_finished():
            # terminal state
            s2 = None
        else:
            raw_state = self._game.get_state()
            self._update_state( raw_state )
            s2 = self._copy_current_state()

        self._transitions.add_transition(s, a, s2, r)

        # Perform q-learning once for a while
        if self._steps % self._update_frequency[0] == 0 and not self.online_mode and self._steps > self._batch_size:
            for i in range(self._update_frequency[1]):
                self._evaluator.learn(self._transitions.get_sample(self._batch_size))
        elif self.online_mode:
            self._evaluator.learn_one(s, a, s2, r)
      
    def run_episode(self, verbose = False):
        self._new_game()
       	if self.learning_mode:
            while not self._game.is_episode_finished():
                self.make_learning_step(verbose)
       	else:
	        while not self._game.is_episode_finished():
	            self.make_step(verbose)

        return np.float32((self._game.get_summary_reward())*self._reward_scale)

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

    def get_network(self):
        return self._evaluator.get_network()