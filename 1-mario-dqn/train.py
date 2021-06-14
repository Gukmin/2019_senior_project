import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))
import random
import tensorflow as tf
import numpy as np

from collections import deque
from nes_py.wrappers import JoypadSpace

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten,Conv2D

from environment.env_supermario import SuperMarioEnv
from environment.actions import SIMPLE_MOVEMENT
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPainter, QBrush, QPen
from PyQt5.QtCore import Qt
import sys


class DQN(tf.keras.Model):
    def __init__(self, action_size, state_size = (1,13,16,4)):
        super(DQN, self).__init__()
        self.conv1 = Conv2D(8, (4,4), strides=(2,2),activation = 'relu',input_shape = state_size)
        self.conv2 = Conv2D(16, (3,3), strides = (1,1),activation = 'relu')
        self.conv3 = Conv2D(16, (2,2), strides = (1,1), activation = 'relu')
        self.flatten = Flatten()
        self. fc = Dense(512, activation='relu')
        self.fc_out = Dense(action_size)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)
        q = self.fc_out(x)
        return q

class DQNAgent:
    def __init__(self, action_size, state_size=(1, 13, 16, 4),render = True):
        self.render = render

        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99
        self.learning_rate = 1e-4
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 200000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end)/self.exploration_steps;
        self.batch_size = 32

        # 리플레이 메모리, 최대 크기 100,000
        self.memory = deque(maxlen=100000)
        self.train_start = 50000
        self.update_target_rate = 10000

        # 모델과 타깃 모델 생성
        self.model = DQN(action_size)
        self.target_model = DQN(action_size)
        self.optimizer = Adam(self.learning_rate)

        # 타깃 모델 초기화
        self.update_target_model()
        self.avg_q_max, self.avg_loss = 0, 0

        self.writer = tf.summary.create_file_writer('summary/newEnvDQN')
        self.model_path = os.path.join(os.getcwd(), 'save_model_newEnvDQN', 'model')

    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model(history)
            return np.argmax(q_value[0])

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))

    # 텐서보드에 학습 정보를 기록
    def draw_tensorboard(self, score, step, episode):
        with self.writer.as_default():
            tf.summary.scalar('Total Reward/Episode', score, step=episode)
            tf.summary.scalar('Average Max Q/Episode',
                              self.avg_q_max / float(step), step=episode)
            tf.summary.scalar('Duration/Episode', step, step=episode)
            tf.summary.scalar('Average Loss/Episode',
                              self.avg_loss / float(step), step=episode)

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출 , 미니배치.
        batch = random.sample(self.memory, self.batch_size)

        history = np.array([sample[0][0] / 255. for sample in batch], dtype=np.float32)
        actions = np.array([sample[1] for sample in batch])
        rewards = np.array([sample[2] for sample in batch])
        next_history = np.array([sample[3][0] / 255. for sample in batch], dtype=np.float32)
        deads = np.array([sample[4] for sample in batch])

        # 학습 파라미터
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            # 예측 = 현재 상태에 대한 모델의 큐함수 Q(St,At,theta)
            predicts = self.model(history)
            one_hot_action = tf.one_hot(actions, self.action_size)
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)

            # 정답 = 다음 상태에 대한 타깃 모델의 큐함수 Q(St+1,a',theta) 최대 값 계산
            target_predicts = self.target_model(next_history)
            target_predicts = tf.stop_gradient(target_predicts) #학습도중 타깃모델학습되지 않도록

            # MSE 계산
            max_q = np.amax(target_predicts, axis=1)
            targets = rewards + (1 - deads) * self.discount_factor * max_q
            loss = tf.reduce_mean(tf.square(targets-predicts))
            self.avg_loss += loss.numpy()

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))

class Window(QMainWindow):
    def __init__(self,game_render = True,pixel_render = True):
        self.pixel_render = pixel_render
        self.game_render = game_render
        if self.pixel_render :
            self.is_None = True
            super().__init__()
            self.title = "SMB pixel"
            self.top = 100
            self.left = 100
            self.width = 170
            self.height =140
            self.InitWindow()
        self.Main()

    def Main(self):
        env = SuperMarioEnv()
        env = JoypadSpace(env,SIMPLE_MOVEMENT)
        agent = DQNAgent(action_size=len(SIMPLE_MOVEMENT),render = self.game_render)

        global_step = 0
        score_avg = 0
        score_max = 0

        num_episode = 5000
        for e in range(num_episode):
            done = False
            dead = False

            step, score = 0,0
            state = env.reset()

            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape([history], (1, 13, 16, 4))

            while not done:
                if agent.render:
                    env.render()
                global_step += 1
                step += 1

                # 바로 전 history를 입력으로 받아 행동을 선택
                action = agent.get_action(history)

                # 선택한 행동으로 환경에서 한 타임스텝 진행
                next_state, reward, done, info = env.step(action) #환경이 observe, reward, 끝났나, 목숨개수 줌

                if self.pixel_render:
                    self.is_None = False
                    self.update()
                    self.tiles = next_state

                next_state = np.reshape([next_state], (1, 13, 16, 1))
                next_history = np.append(next_state, history[:, :, :, :3], axis=3)

                agent.avg_q_max += np.amax(agent.model(np.float32(history / 255.))[0])

                score += reward

                # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장 후 학습
                agent.append_sample(history, action, reward, next_history, info['dead'])

                # 리플레이 메모리 크기가 정해놓은 수치에 도달한 시점부터 모델 학습 시작
                if len(agent.memory) >= agent.train_start:
                    agent.train_model()
                    # 일정 시간마다 타겟모델을 모델의 가중치로 업데이트
                    if global_step % agent.update_target_rate == 0:
                        agent.update_target_model()
                history = next_history
                if done:
                    # 각 에피소드 당 학습 정보를 기록
                    if global_step > agent.train_start:
                        agent.draw_tensorboard(score, step, e)

                    score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                    score_max = score if score > score_max else score_max

                    log = "episode: {:5d} | ".format(e)
                    log += "score: {:4.1f} | ".format(score)
                    log += "score max : {:4.1f} | ".format(score_max)
                    log += "score avg: {:4.1f} | ".format(score_avg)
                    log += "memory length: {:5d} | ".format(len(agent.memory))
                    log += "epsilon: {:.3f} | ".format(agent.epsilon)
                    log += "q avg : {:3.2f} | ".format(agent.avg_q_max / float(step))
                    log += "avg loss : {:3.2f}".format(agent.avg_loss / float(step))
                    print(log)

                    agent.avg_q_max, agent.avg_loss = 0, 0

            # 50 에피소드마다 모델 저장
            if e % 50 == 0:
                agent.model.save_weights("./save_model_newEnvDQN/model", save_format="tf")

    def InitWindow(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.top,self.left,self.width,self.height)
        self.show()

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)

        self.draw_state(painter)
        painter.end()

    def draw_state(self, painter: QPainter):
        if self.is_None:
            return
        for row in range(13):
            for col in range(16):
                painter.setPen(QPen(Qt.black,  1, Qt.SolidLine))
                painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))
                x_start = 5 + (10 * col)
                y_start = 5 + (10 * row)
                if self.tiles[row][col] == 85:
                    painter.setBrush(QBrush(Qt.red,Qt.SolidPattern))
                elif self.tiles[row][col] == 190:
                    painter.setBrush(QBrush(Qt.blue,Qt.SolidPattern))
                elif self.tiles[row][col] == 255:
                    painter.setBrush(QBrush(Qt.green,Qt.SolidPattern))
                painter.drawRect(x_start, y_start, 10, 10)
#main
if __name__ == "__main__":
    App = QApplication(sys.argv)
    window = Window(game_render = False,pixel_render=False)
    sys.exit(App.exec())


