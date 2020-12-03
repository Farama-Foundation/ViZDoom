from __future__ import print_function
import vizdoom as vzd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import itertools as it
import skimage.transform

from vizdoom import GameVariable, Mode
from time import sleep
from matplotlib import pyplot as plt
from collections import deque
from tqdm import trange

# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device('cpu')


def preprocess(img, resolution=(30, 45)):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

def create_simple_game():
  game = vzd.DoomGame()
  game.load_config("../../scenarios/simpler_basic.cfg")
  game.set_window_visible(False)
  game.set_mode(Mode.PLAYER)
  game.init()

  return game

def run(game, agent, actions, num_episodes, frame_repeat,
        steps_per_episode=2000):
    scores = []

    for episode in range(num_episodes):
        game.new_episode()
        train_scores = []
        global_step = 0
        done = False
        print("Episode #" + str(episode + 1))
        print("Epsilon " + str(agent.epsilon))

        for _ in trange(steps_per_episode, leave=False):
            state = preprocess(game.get_state().screen_buffer)
            action = agent.get_action(state)
            reward = game.make_action(actions[action], frame_repeat)
            done = game.is_episode_finished()

            if not done:
              next_state = preprocess(game.get_state().screen_buffer)
            else:
              next_state = np.zeros((1, 30, 45)).astype(np.float32)

            agent.append_memory(state, action, reward, next_state, done)

            if global_step > agent.batch_size:
                agent.train()

            if done:
              scores.append(game.get_total_reward())
              train_scores.append(game.get_total_reward())
              game.new_episode()

            global_step += 1

        train_scores = np.array(train_scores)

        print("Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()), \
              "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

    game.close()
    return agent, game

class TestNet(nn.Module):
     def __init__(self, available_actions_count):
         super(TestNet, self).__init__()
         self.conv1 = nn.Conv2d(1, 8, kernel_size=6, stride=3)
         self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
         self.fc1 = nn.Linear(192, 128)
         self.fc2 = nn.Linear(128, available_actions_count)

     def forward(self, x):
         x = torch.from_numpy(x).float().to(DEVICE)
         x = F.relu(self.conv1(x))
         x = F.relu(self.conv2(x))
         x = x.view(-1, 192)
         x = F.relu(self.fc1(x))
         return self.fc2(x)

class DQNAgent:
    def __init__(self, action_size, epsilon=1, memory_size=10000,
                 batch_size=64, discount_factor=0.99, lr=25e-5, epsilon_decay=0.9996,
                 epsilon_min=0.1):
        self.action_size = action_size
        self.q_net = TestNet(action_size).to(DEVICE)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.discount = discount_factor
        self.lr = lr
        self.memory = deque(maxlen=memory_size)
        self.opt = optim.SGD(self.q_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            state = np.expand_dims(state, axis=0)
            action = torch.argmax(self.q_net(state)).item()
            return action

    def append_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        batch = random.sample(self.memory, self.batch_size)
        batch = np.array(batch, dtype=object)

        states = np.stack(batch[:,0]).astype(float)
        actions = batch[:,1].astype(int)
        rewards = batch[:,2].astype(float)
        next_states = np.stack(batch[:,3]).astype(float)
        dones = batch[:,4].astype(bool)
        not_dones = ~dones
        
        # value of the next states
        next_state_values = np.max(self.q_net(next_states).data.numpy(), 1)
        next_state_values = next_state_values[not_dones]

        Y = rewards.copy()
        Y[not_dones] += self.discount * next_state_values
        Y = torch.from_numpy(Y).float()

        # this selects only the q values of the actions taken
        state_values = self.q_net(states)[np.arange(self.batch_size), actions].float()

        self.opt.zero_grad()
        loss = self.criterion(Y, state_values)
        loss.backward()
        self.opt.step()

        if self.epsilon > self.epsilon_min:
          self.epsilon *= self.epsilon_decay

        else:
          self.epsilon = self.epsilon_min

if __name__=='__main__':
    train_episodes = 5
    episodes_to_watch = 10
    lr = 0.00025
    frame_repeat = 12

    actions = [[True, False, False], [False, True, False], [False, False, True]]
    game = create_simple_game()
    agent = DQNAgent(len(actions))
    agent, game = run(game, agent, actions, num_episodes=train_episodes, frame_repeat=frame_repeat)

    print("======================================")
    print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
    game.close()
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()

    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = agent.get_action(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)
