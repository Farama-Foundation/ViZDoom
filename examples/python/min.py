from vizdoom import *
import random
import time

game = DoomGame()
game.load_config("../config/basic.cfg")
game.init()

shoot = [0,0,1]
left = [1,0,0]
right = [0,1,0]
actions = [shoot, left, right]

episodes = 10
for i in range(episodes):
    game.new_episode()
    while not game.is_episode_finished():
    	state = game.get_state()
        img = state.image_buffer
        misc = state.game_variables
        reward = game.make_action(random.choice(actions))
        time.sleep(0.028)
    print "Result:", game.get_summary_reward()