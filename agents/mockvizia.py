import numpy as np
import random


class MockDoomGame:
    def __init__(self):        
        self._x = 40
        self._y = 30
        self._offset = 0.0
        self._living_reward = -1.0
        self._miss_penalty = 5.0
        self._hit_reward = 101.0
        self._timeout = 200
        self._random_background = False
        self._noise_level = 0.0

        self._action_format = 3
        self._state = None
        self._finished = True
        self._summary_reward = 0
        self._state_num = 0
        self._channels = 1
        self._no_shooting = 8
        self._no_shooting_count = self._no_shooting

    def set_living_reward(self, reward):
        self._living_reward = reward

    def set_timeout(self, timeout):
        self._timeout = timeout
    
    def set_screen_resolution(self, x, y):
        self._x = max(1,x)
        self._y = max(1, y)
    
    def get_screen_width(self):
        return self._x

    def get_screen_channels(self):
        return self._channels

    def get_screen_height(self):
        return self._y

    def init(self):
        
        target_x = max(1, int(self._x/10))
        target_y = max(1, int(self._y/10))
        self._state_template = np.zeros([self._x, self._y], dtype=np.uint8)
        #self._state_template.fill(0.3)
        self._left_space = int((self._x - target_x)/2)
        self._right_space = self._x - self._left_space - target_x
        up_space = int((self._y - target_y)/2)
        down_space = self._y - up_space - target_y
        self._state_template[self._left_space:self._x - self._right_space, up_space:self._y - down_space] = 255
        
        self.new_episode()


    def is_episode_finished(self):
        return self._finished


    def get_action_format(self):
        return self._action_format

    def get_summary_reward(self):
        return self._summary_reward

    def new_episode(self):
        self._finished = False
        self._state_num = 0
        self._summary_reward = 0

        #self._state = self._dtype(np.random.rand(self._y, self._x)) * self._noise_level
        #self._state[self._aimY] = 0.0
       
        self._offset = random.randint(-self._left_space, self._right_space)
        self._state = np.roll(self._state_template, self._offset,0)
        self._no_shooting_count = self._no_shooting

        
    def make_action(self, action):

        if self._finished:
            print "Making action in a finished game."
            return None
        else:
            reward = self._living_reward
           

            # shoot
            if action[2] and self._no_shooting_count == 0 :
                if self._state[ max(0, self._x/2 -1), max(0,self._y/2-1)] >0:
                    reward += self._hit_reward
                    self._finished = True
                    self._state = None
                else:
                    reward -= self._miss_penalty
                self._no_shooting_count = self._no_shooting
            else:
                self._no_shooting_count = max(0,self._no_shooting_count-1)

            # right
            if not self._finished:
                if action[0] and not action[1]:
                    if self._offset > -self._left_space:
                        self._offset -= 1
                        self._state = np.roll(self._state, -1, axis = 0)
                # left
                elif action[1] and not action[0]:
                    if self._offset < self._right_space :
                        self._offset += 1
                        self._state = np.roll(self._state, 1, axis = 0)
                

            
                
                    

            self._summary_reward += reward

            self._state_num += 1

            if self._state_num >= self._timeout:
                self._finished = True
                self._state = None

            return reward

    def get_state(self):
        if self._state is None:
            return None
        else:
            ret_val = (self._state_num, self._state.copy())

            return ret_val

    def get_game_var_len(self):
        return 0

    def set_no_shooting_time(self, steps):
        self._no_shooting_count = steps
        self._no_shooting = steps

    def set_miss_penalty(self, penalty):
        self._miss_penalty = penalty