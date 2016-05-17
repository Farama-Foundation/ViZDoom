#!/usr/bin/python

import itertools as it
import pickle
from random import sample, randint, random
from time import time
from vizdoom import *

import cv2
import numpy as np
import theano
from lasagne.init import GlorotUniform, Constant
from lasagne.layers import Conv2DLayer, InputLayer, DenseLayer, MaxPool2DLayer, get_output, get_all_params, \
    get_all_param_values, set_all_param_values
from lasagne.nonlinearities import rectify
from lasagne.objectives import squared_error
from lasagne.updates import rmsprop
from theano import tensor
from tqdm import *
from time import sleep

# Q-learning settings:
replay_memory_size = 10000
discount_factor = 0.99
start_epsilon = float(1.0)
end_epsilon = float(0.1)
epsilon = start_epsilon
static_epsilon_steps = 5000
epsilon_decay_steps = 20000
epsilon_decay_stride = (start_epsilon - end_epsilon) / epsilon_decay_steps

# Max reward is about 100 (for killing) so it'll be normalized
reward_scale = 0.01

# Some of the network's and learning settings:
learning_rate = 0.00001
batch_size = 32
epochs = 20
training_steps_per_epoch = 5000
test_episodes_per_epoch = 100

# Other parameters
skiprate = 7
downsampled_x = 60
downsampled_y = int(2/3.0*downsampled_x)
episodes_to_watch = 10

# Where to save and load network's weights.
params_savefile = "basic_params"
params_loadfile = None

# Function for converting images
def convert(img):
    img = img[0].astype(np.float32) / 255.0
    img = cv2.resize(img, (downsampled_x, downsampled_y))
    return img


# Replay memory:
class ReplayMemory:
    def __init__(self, capacity):

        state_shape = (capacity, 1, downsampled_y, downsampled_x)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.nonterminal = np.zeros(capacity, dtype=np.bool_)

        self.size = 0
        self.capacity = capacity
        self.oldest_index = 0

    def add_transition(self, s1, action, s2, reward):
        self.s1[self.oldest_index, 0] = s1
        if s2 is None:
            self.nonterminal[self.oldest_index] = False
        else:
            self.s2[self.oldest_index, 0] = s2
            self.nonterminal[self.oldest_index] = True
        self.a[self.oldest_index] = action
        self.r[self.oldest_index] = reward

        self.oldest_index = (self.oldest_index + 1) % self.capacity

        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.s2[i], self.a[i], self.r[i], self.nonterminal[i]


# Creates the network:
def create_network(available_actions_num):
    # Creates the input variables
    s1 = tensor.tensor4("States")
    a = tensor.vector("Actions", dtype="int32")
    q2 = tensor.vector("Next State best Q-Value")
    r = tensor.vector("Rewards")
    nonterminal = tensor.vector("Nonterminal", dtype="int8")

    # Creates the input layer of the network.
    dqn = InputLayer(shape=[None, 1, downsampled_y, downsampled_x], input_var=s1)

    # Adds 3 convolutional layers, each followed by a max pooling layer.
    dqn = Conv2DLayer(dqn, num_filters=32, filter_size=[8, 8],
                      nonlinearity=rectify, W=GlorotUniform("relu"),
                      b=Constant(.1))
    dqn = MaxPool2DLayer(dqn, pool_size=[2, 2])
    dqn = Conv2DLayer(dqn, num_filters=64, filter_size=[4, 4],
                      nonlinearity=rectify, W=GlorotUniform("relu"),
                      b=Constant(.1))

    dqn = MaxPool2DLayer(dqn, pool_size=[2, 2])
    dqn = Conv2DLayer(dqn, num_filters=64, filter_size=[3, 3],
                      nonlinearity=rectify, W=GlorotUniform("relu"),
                      b=Constant(.1))
    dqn = MaxPool2DLayer(dqn, pool_size=[2, 2])
    # Adds a single fully connected layer.
    dqn = DenseLayer(dqn, num_units=512, nonlinearity=rectify, W=GlorotUniform("relu"),
                     b=Constant(.1))

    # Adds a single fully connected layer which is the output layer.
    # (no nonlinearity as it is for approximating an arbitrary real function)
    dqn = DenseLayer(dqn, num_units=available_actions_num, nonlinearity=None)

    # Theano stuff
    q = get_output(dqn)
    # Only q for the chosen actions is updated more or less according to following formula:
    # target Q(s,a,t) = r + gamma * max Q(s2,_,t+1)
    target_q = tensor.set_subtensor(q[tensor.arange(q.shape[0]), a], r + discount_factor * nonterminal * q2)
    loss = squared_error(q, target_q).mean()

    # Updates the parameters according to the computed gradient using rmsprop.
    params = get_all_params(dqn, trainable=True)
    updates = rmsprop(loss, params, learning_rate)

    # Compiles theano functions
    print "Compiling the network ..."
    function_learn = theano.function([s1, q2, a, r, nonterminal], loss, updates=updates, name="learn_fn")
    function_get_q_values = theano.function([s1], q, name="eval_fn")
    function_get_best_action = theano.function([s1], tensor.argmax(q), name="test_fn")
    print "Network compiled."

    # Returns Theano objects for the net and functions.
    # We wouldn't need the net anymore but it is nice to save your model.
    return dqn, function_learn, function_get_q_values, function_get_best_action


# Creates and initializes the environment.
print "Initializing doom..."
game = DoomGame()
game.load_config("../../examples/config/learning.cfg")
game.init()
print "Doom initialized."

# Creates all possible actions.
n = game.get_available_buttons_size()
actions = []
for perm in it.product([0, 1], repeat=n):
    actions.append(list(perm))

# Creates replay memory which will store the transitions
memory = ReplayMemory(capacity=replay_memory_size)
net, learn, get_q_values, get_best_action = create_network(len(actions))

# Loads the  network's parameters if the loadfile was specified
if params_loadfile is not None:
    params = pickle.load(open(params_loadfile, "r"))
    set_all_param_values(net, params)


# Makes an action according to epsilon greedy policy and performs a single backpropagation on the network.
def perform_learning_step():
    # Checks the state and downsamples it.
    s1 = convert(game.get_state().image_buffer)

    # With probability epsilon makes a random action.
    if random() <= epsilon:
        a = randint(0, len(actions) - 1)
    else:
        # Chooses the best action according to the network.
        a = get_best_action(s1.reshape([1, 1, downsampled_y, downsampled_x]))
    reward = game.make_action(actions[a], skiprate + 1)
    reward *= reward_scale

    if game.is_episode_finished():
        s2 = None
    else:
        s2 = convert(game.get_state().image_buffer)
    # Remember the transition that was just experienced.
    memory.add_transition(s1, a, s2, reward)

    # Gets a single, random minibatch from the replay memory and learns from it.
    if memory.size > batch_size:
        s1, s2, a, reward, nonterminal = memory.get_sample(batch_size)
        q2 = np.max(get_q_values(s2), axis=1)
        loss = learn(s1, q2, a, reward, nonterminal)
    else:
        loss = 0
    return loss


print "Starting the training!"

steps = 0
for epoch in range(epochs):
    print "\nEpoch", epoch
    train_time = 0
    train_episodes_finished = 0
    train_loss = []
    train_rewards = []

    train_start = time()
    print "\nTraining ..."
    game.new_episode()
    for learning_step in tqdm(range(training_steps_per_epoch)):
        # Learning and action is here.
        train_loss.append(perform_learning_step())
        # I
        if game.is_episode_finished():
            r = game.get_total_reward()
            train_rewards.append(r)
            game.new_episode()
            train_episodes_finished += 1

        steps += 1
        if steps > static_epsilon_steps:
            epsilon = max(end_epsilon, epsilon - epsilon_decay_stride)

    train_end = time()
    train_time = train_end - train_start
    mean_loss = np.mean(train_loss)

    print train_episodes_finished, "training episodes played."
    print "Training results:"

    train_rewards = np.array(train_rewards)

    print "mean:", train_rewards.mean(), "std:", train_rewards.std(), "max:", train_rewards.max(), "min:", train_rewards.min(), "mean_loss:", mean_loss, "epsilon:", epsilon
    print "t:", str(round(train_time, 2)) + "s"

    # Testing
    test_episode = []
    test_rewards = []
    test_start = time()
    print "Testing..."
    for test_episode in tqdm(range(test_episodes_per_epoch)):
        game.new_episode()
        while not game.is_episode_finished():
            state = convert(game.get_state().image_buffer).reshape([1, 1, downsampled_y, downsampled_x])
            best_action_index = get_best_action(state)

            game.make_action(actions[best_action_index], skiprate + 1)
        r = game.get_total_reward()
        test_rewards.append(r)

    test_end = time()
    test_time = test_end - test_start
    print "Test results:"
    test_rewards = np.array(test_rewards)
    print "mean:", test_rewards.mean(), "std:", test_rewards.std(), "max:", test_rewards.max(), "min:", test_rewards.min()
    print "t:", str(round(test_time, 2)) + "s"

    if params_savefile:
        print "Saving network weigths to:", params_savefile
        pickle.dump(get_all_param_values(net), open(params_savefile, "w"))
    print "========================="

print "Training finished! Time to watch!"

game.close()
game.set_window_visible(True)
game.set_mode(Mode.ASYNC_PLAYER)
game.init()

# Sleeping time between episodes, for convenience.
episode_sleep = 0.5

for i in range(episodes_to_watch):
    game.new_episode()
    while not game.is_episode_finished():
        state = convert(game.get_state().image_buffer).reshape([1, 1, downsampled_y, downsampled_x])
        best_action_index = get_best_action(state)
        game.set_action(actions[best_action_index])
        for i in range(skiprate+1):
            game.advance_action()

    sleep(episode_sleep)
    r = game.get_total_reward()
    print "Total reward: ", r
