import os
import sys
import time
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))

import threading
import numpy as np
import tensorflow as tf

from tensorflow.keras.initializers import RandomUniform
from tensorflow.compat.v1.train import AdamOptimizer
from tensorflow.keras.layers import Flatten, Dense,Conv2D

from nes_py.wrappers import JoypadSpace
from environment.env_supermario import SuperMarioEnv
from environment.actions import SIMPLE_MOVEMENT

# 글로벌 변수
global episode, score_avg, score_max
episode, score_avg, score_max = 0, 0, 0
num_episode = 10000

# ActorCritic 인공신경망
class ActorCritic(tf.keras.Model):
    def __init__(self, action_size,state_size = (1,13,16,4)):
        super(ActorCritic, self).__init__()
        #정책신경망과 가치신경망의 모델을 공유
        self.conv1 = Conv2D(8, (4, 4), strides=(2, 2), activation='relu', input_shape=state_size)
        self.conv2 = Conv2D(16, (3, 3), strides=(1, 1), activation='relu')
        self.conv3 = Conv2D(16, (2, 2), strides=(1, 1), activation='relu')
        self.flatten = Flatten()
        self.shared_fc = Dense(512, activation='relu')
        self.policy = Dense(action_size,activation = 'linear')
        self.value = Dense(1,activation = 'linear')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten
        x = self.shared_fc(x)
        policy = self.policy(x)
        value = self.value(x)

        return policy, value

#글로벌 신경망
class A3CAgent():
    def __init__(self, action_size):
        self.state_size = (1,13, 16, 4)
        self.action_size = action_size
        self.discount_factor = 0.99
        self.lr = 1e-4
        self.threads = 8

        self.global_model = ActorCritic(self.action_size)
        self.global_model.build(tf.TensorShape((None, *self.state_size)))

        self.optimizer = AdamOptimizer(self.lr, use_locking=True)

        self.writer = tf.summary.create_file_writer('summary/newEnvA3C')
        self.model_path = os.path.join(os.getcwd(), 'save_model_newEnvA3C', 'model')

    # 쓰레드를 만들어 학습을 하는 함수
    def train(self):
        # 쓰레드 수 만큼 Runner 클래스 생성
        runners = [Runner(self.action_size, self.state_size,
                          self.global_model, self.optimizer,
                          self.discount_factor,
                          self.writer) for i in range(self.threads)]

        for i, runner in enumerate(runners):
            runner.start()

        while True:
            self.global_model.save_weights(self.model_path, save_format="tf")
            time.sleep(600)


class Runner(threading.Thread):
    global_episode = 0

    def __init__(self, action_size, state_size, global_model,
                 optimizer, discount_factor, writer):
        threading.Thread.__init__(self)

        self.action_size = action_size
        self.state_size = state_size
        self.global_model = global_model
        self.optimizer = optimizer
        self.discount_factor = discount_factor

        self.states, self.actions, self.rewards = [], [], []

        # 환경, 로컬신경망, 텐서보드 생성
        self.local_model = ActorCritic(action_size)
        self.env = SuperMarioEnv()
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        self.writer = writer

        # 학습 정보를 기록할 변수
        self.avg_p_max = 0
        self.avg_loss = 0
        # k-타임스텝 값 설정
        self.t_max = 20
        self.t = 0

    # 텐서보드에 학습 정보를 기록
    def draw_tensorboard(self, score, step, e):
        avg_p_max = self.avg_p_max / float(step)
        with self.writer.as_default():
            tf.summary.scalar('Total Reward/Episode', score, step=e)
            tf.summary.scalar('Average Max Prob/Episode', avg_p_max, step=e)
            tf.summary.scalar('Duration/Episode', step, step=e)

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, history):
        history = np.float32(history / 255.)
        policy = self.local_model(history)[0][0]
        policy = tf.nn.softmax(policy)
        action_index = np.random.choice(self.action_size, 1, p=policy.numpy())[0]
        return action_index, policy

    # 샘플을 저장
    def append_sample(self, history, action, reward):
        self.states.append(history)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

    # k-타임스텝의 prediction 계산
    def discounted_prediction(self, rewards, done):
        discounted_prediction = np.zeros_like(rewards)
        running_add = 0

        if not done:
            # 가치함수
            last_state = np.float32(self.states[-1] / 255.)
            running_add = self.local_model(last_state)[-1][0].numpy()

        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_prediction[t] = running_add
        return discounted_prediction

    # 저장된 샘플들로 A3C의 오류함수를 계산
    def compute_loss(self, done):

        discounted_prediction = self.discounted_prediction(self.rewards, done)
        discounted_prediction = tf.convert_to_tensor(discounted_prediction[:, None],
                                                     dtype=tf.float32)

        states = np.zeros((len(self.states), 13, 16, 4))

        for i in range(len(self.states)):
            states[i] = self.states[i]
        states = np.float32(states / 255.)

        policy, values = self.local_model(states)

        # 가치 신경망 업데이트
        advantages = discounted_prediction - values
        critic_loss = 0.5 * tf.reduce_sum(tf.square(advantages))

        # 정책 신경망 업데이트
        action = tf.convert_to_tensor(self.actions, dtype=tf.float32)
        policy_prob = tf.nn.softmax(policy)
        action_prob = tf.reduce_sum(action * policy_prob, axis=1, keepdims=True)
        cross_entropy = - tf.math.log(action_prob + 1e-10)
        actor_loss = tf.reduce_sum(cross_entropy * tf.stop_gradient(advantages))

        entropy = tf.reduce_sum(policy_prob * tf.math.log(policy_prob + 1e-10), axis=1)
        entropy = tf.reduce_sum(entropy)
        actor_loss += 0.01 * entropy

        total_loss = 0.5 * critic_loss + actor_loss

        return total_loss

    # 로컬신경망을 통해 그레이디언트를 계산하고, 글로벌 신경망을 계산된 그레이디언트로 업데이트
    def train_model(self, done):

        global_params = self.global_model.trainable_variables
        local_params = self.local_model.trainable_variables

        with tf.GradientTape() as tape:
            total_loss = self.compute_loss(done)

        grads = tape.gradient(total_loss, local_params)
        grads, _ = tf.clip_by_global_norm(grads,40.0)

        self.optimizer.apply_gradients(zip(grads, global_params))
        self.local_model.set_weights(self.global_model.get_weights())
        self.states, self.actions, self.rewards = [], [], []

    def run(self):
        # 액터러너끼리 공유해야하는 글로벌 변수
        global episode, score_avg, score_max

        step = 0
        while episode < num_episode:
            done = False
            dead = False

            score = 0
            state = self.env.reset()
            history = np.stack([state, state, state, state], axis=2)
            history = np.reshape([history], (1, 13, 16, 4))

            while not done:
                step += 1
                self.t += 1

                # 정책 확률에 따라 행동을 선택
                action, policy = self.get_action(history)
                next_state, reward, done, info = self.env.step(action)

                next_state = np.reshape([next_state], (1, 13, 16, 1))
                next_history = np.append(next_state, history[:, :, :, :3], axis=3)

                self.avg_p_max += np.amax(policy.numpy())

                score += reward

                # 샘플을 저장
                self.append_sample(history, action, reward)
                history = next_history

                # 에피소드가 끝나거나 최대 타임스텝 수에 도달하면 학습을 진행
                if self.t >= self.t_max or done:
                    self.train_model(done)
                    self.t = 0

                if done:
                    # 각 에피소드 당 학습 정보를 기록
                    episode += 1
                    score_max = score if score > score_max else score_max
                    score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score

                    log = "episode: {:5d} | score : {:4.1f} | ".format(episode, score)
                    log += "score max : {:4.1f} | ".format(score_max)
                    log += "score avg : {:.3f}".format(score_avg)
                    print(log)

                    self.draw_tensorboard(score, step, episode)

                    self.avg_p_max = 0
                    step = 0

if __name__ == "__main__":
    global_agent = A3CAgent(action_size=len(SIMPLE_MOVEMENT))
    global_agent.train()
