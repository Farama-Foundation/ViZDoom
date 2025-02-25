#!/usr/bin/env python3

# M. Kempka, T.Sternal, M.Wydmuch, Z.Boztoprak
# January 2021

import itertools as it
import os
from collections import deque
from random import sample
from time import sleep, time

import numpy as np
import skimage.color
import skimage.transform
import tensorflow as tf
from setuptools.command.setopt import config_file
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Flatten, ReLU
from tensorflow.keras.optimizers import SGD
from tqdm import trange

import vizdoom as vzd
from vizdoom import scenarios_path

tf.compat.v1.enable_eager_execution()
tf.executing_eagerly()

# Q-learning settings
learning_rate = 0.00025
discount_factor = 0.99
replay_memory_size = 10000
num_train_epochs = 20
learning_steps_per_epoch = 2000
target_net_update_steps = 1000

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frames_per_action = 4
resolution = (30, 45)
episodes_to_watch = 20

save_model = True
load = False
skip_learning = False
watch = True

# Configuration file path
# 설정 파일과 모델 저장 경로 설정
# config_file_path = os.path.join(vzd.scenarios_path, "deadly_corridor.cfg")
config_file_path = r"C:\Users\schale\ViZDoom\scenarios\deadly_corridor.cfg"
model_savefolder = "./model"

# GPU 사용 가능 여부 확인 후 모델 학습에 사용할 디바이스 설정
if len(tf.config.experimental.list_physical_devices("GPU")) > 0:
    print("GPU available")
    DEVICE = "/gpu:0"
else:
    print("No GPU available")
    DEVICE = "/cpu:0"


# 이미지를 resolution 변수에 맞게 resize
def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=-1)

    return tf.stack(img)


# 게임 설정 및 초기화
def initialize_game():
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")

    return game


# 강화 학습 에이전트 정의
class DQNAgent:
    def __init__(
        self, num_actions=8, epsilon=1, epsilon_min=0.1, epsilon_decay=0.9995, load=load # 액션 수, 엡실론
    ):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount_factor = discount_factor
        self.num_actions = num_actions
        self.optimizer = SGD(learning_rate)

        if load: # 로드 할 모델이 있는 경우
            print("Loading model from: ", model_savefolder)
            self.dqn = tf.keras.models.load_model(model_savefolder)
        else: # 없는 경우
            self.dqn = DQN(self.num_actions)
            self.target_net = DQN(self.num_actions)

    def update_target_net(self):
        self.target_net.set_weights(self.dqn.get_weights())

    def choose_action(self, state): # state를 입력으로 액션 선택
        if self.epsilon < np.random.uniform(0, 1):
            action = int(tf.argmax(self.dqn(tf.reshape(state, (1, 30, 45, 1))), axis=1))
        else:
            action = np.random.choice(range(self.num_actions), 1)[0]

        return action

    def train_dqn(self, samples):
        screen_buf, actions, rewards, next_screen_buf, dones = split_tuple(samples)

        row_ids = list(range(screen_buf.shape[0]))

        ids = extractDigits(row_ids, actions)
        done_ids = extractDigits(np.where(dones)[0])

        with tf.GradientTape() as tape:
            tape.watch(self.dqn.trainable_variables)

            Q_prev = tf.gather_nd(self.dqn(screen_buf), ids)

            Q_next = self.target_net(next_screen_buf)
            Q_next = tf.gather_nd(
                Q_next,
                extractDigits(row_ids, tf.argmax(agent.dqn(next_screen_buf), axis=1)),
            )

            q_target = rewards + self.discount_factor * Q_next

            if len(done_ids) > 0:
                done_rewards = tf.gather_nd(rewards, done_ids)
                q_target = tf.tensor_scatter_nd_update(
                    tensor=q_target, indices=done_ids, updates=done_rewards
                )

            td_error = tf.keras.losses.MSE(q_target, Q_prev)

        gradients = tape.gradient(td_error, self.dqn.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.dqn.trainable_variables))

        if self.epsilon > self.epsilon_min: # epsilon이 최솟값보다 클 때, epsilon_decay를 곱함
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min


def split_tuple(samples):
    samples = np.array(samples, dtype=object)
    screen_buf = tf.stack(samples[:, 0])
    actions = samples[:, 1]
    rewards = tf.stack(samples[:, 2])
    next_screen_buf = tf.stack(samples[:, 3])
    dones = tf.stack(samples[:, 4])
    return screen_buf, actions, rewards, next_screen_buf, dones


def extractDigits(*argv):
    if len(argv) == 1:
        return list(map(lambda x: [x], argv[0]))

    return list(map(lambda x, y: [x, y], argv[0], argv[1]))


def get_samples(memory):
    if len(memory) < batch_size:
        sample_size = len(memory)
    else:
        sample_size = batch_size

    return sample(memory, sample_size)

def calculate_reward(game, action, prev_state):
    # 기본 보상
    reward = game.make_action(actions[action], frames_per_action)

    if game.get_state():
        game_vars = game.get_state().game_variables
        ammo = game_vars[0]
        health = game_vars[1]
        killcount = game_vars[2]

        # 플레이어의 좌표 정보 (게임 상태에서 추출)
        player_x, player_y = game_vars[3], game_vars[4]

        # Vest(방탄복) 좌표 (미리 지정된 값, 맵을 분석해서 가져와야 함)
        vest_x, vest_y = 1300, 0
        # 1300, [-16,16]
        # vest_distance = ((player_x - vest_x) ** 2 + (player_y - vest_y) ** 2) ** 0.5
        vest_distance = (player_x - vest_x) ** 2 ** 0.5

        # 🔹 탄약 감소 시 패널티
        if ammo < prev_state["ammo"]:
            reward -= 2

        # 🔹 적을 처치하면 보상 증가
        if killcount > prev_state["killcount"]:
            reward += 10

        # 🔹 체력이 줄어들면 패널티
        if health < prev_state["health"]:
            reward -= 5

        # 🔹 Vest(방탄복)에 가까워지면 보상 증가
        if vest_distance < prev_state["vest_distance"]:
            reward += 5

        # 🔹 prev_state 업데이트 (다음 프레임에서 사용)
        prev_state["ammo"] = ammo
        prev_state["killcount"] = killcount
        prev_state["health"] = health
        prev_state["vest_distance"] = vest_distance

    return reward * 0.1



def run(agent, game, replay_memory):
    time_start = time()

    for episode in range(num_train_epochs):
        train_scores = []
        print("\nEpoch %d\n-------" % (episode + 1))

        game.new_episode()
        prev_state = {"ammo": 52, "health": 100, "killcount": 0, "vest_distance": float("inf")}
        total_reward = 0
        for i in trange(learning_steps_per_epoch, leave=False):
            state = game.get_state()
            screen_buf = preprocess(state.screen_buffer) # 전처리 된 게임 이미지
            action = agent.choose_action(screen_buf) # screen_buf로 액션 결정
            # reward = game.make_action(actions[action], frames_per_action)
            reward = calculate_reward(game, action, prev_state)
            done = game.is_episode_finished()

            if not done:
                next_screen_buf = preprocess(game.get_state().screen_buffer)
            else:
                next_screen_buf = tf.zeros(shape=screen_buf.shape)

            if done:
                train_scores.append(game.get_total_reward())

                game.new_episode()

            replay_memory.append((screen_buf, action, reward, next_screen_buf, done))

            if i >= batch_size:
                agent.train_dqn(get_samples(replay_memory))

            if (i % target_net_update_steps) == 0:
                agent.update_target_net()

            total_reward += reward

        train_scores = np.array(train_scores)
        print(
            "Results: mean: {:.1f}±{:.1f},".format(
                train_scores.mean(), train_scores.std()
            ),
            "min: %.1f," % train_scores.min(),
            "max: %.1f," % train_scores.max(),
        )

        test(test_episodes_per_epoch, game, agent)
        print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))
        print(f"Total reward: {total_reward}")


def test(test_episodes_per_epoch, game, agent):
    test_scores = []

    print("\nTesting...")
    for test_episode in trange(test_episodes_per_epoch, leave=False):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = agent.choose_action(state)
            game.make_action(actions[best_action_index], frames_per_action)

        r = game.get_total_reward()
        test_scores.append(r)

    test_scores = np.array(test_scores)
    print(
        f"Results: mean: {test_scores.mean():.1f}±{test_scores.std():.1f},",
        "min: %.1f" % test_scores.min(),
        "max: %.1f" % test_scores.max(),
    )


class DQN(Model):
    def __init__(self, num_actions):
        super().__init__()
        self.conv1 = Sequential(
            [
                Conv2D(8, kernel_size=6, strides=3, input_shape=(30, 45, 1)),
                BatchNormalization(),
                ReLU(),
            ]
        )

        self.conv2 = Sequential(
            [
                Conv2D(8, kernel_size=3, strides=2, input_shape=(9, 14, 8)),
                BatchNormalization(),
                ReLU(),
            ]
        )

        self.flatten = Flatten()

        self.state_value = Dense(1)
        self.advantage = Dense(num_actions)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x1 = x[:, :96]
        x2 = x[:, 96:]
        x1 = self.state_value(x1)
        x2 = self.advantage(x2)

        x = x1 + (x2 - tf.reshape(tf.math.reduce_mean(x2, axis=1), shape=(-1, 1)))
        return x


if __name__ == "__main__":
    agent = DQNAgent()
    game = initialize_game()
    replay_memory = deque(maxlen=replay_memory_size)

    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    with tf.device(DEVICE):

        if not skip_learning:
            print("Starting the training!")

            run(agent, game, replay_memory)

            game.close()
            print("======================================")
            print("Training is finished.")

            if save_model:
                agent.dqn.save(model_savefolder)

        game.close()

        if watch:
            game.set_window_visible(True)
            game.set_mode(vzd.Mode.ASYNC_PLAYER)
            game.init()

            for _ in range(episodes_to_watch):
                game.new_episode()
                while not game.is_episode_finished():
                    state = preprocess(game.get_state().screen_buffer)
                    best_action_index = agent.choose_action(state)

                    # Instead of make_action(a, frame_repeat) in order to make the animation smooth
                    game.set_action(actions[best_action_index])
                    for _ in range(frames_per_action):
                        game.advance_action()

                # Sleep between episodes
                sleep(1.0)
                score = game.get_total_reward()
                print("Total score: ", score)
