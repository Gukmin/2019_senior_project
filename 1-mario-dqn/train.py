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
        self.conv2 = Conv2D(16, (2,2), strides = (1,1),activation = 'relu')
        self.flatten = Flatten()
        self. fc = Dense(128, activation='relu')
        self.fc_out = Dense(action_size)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)
        q = self.fc_out(x)
        return q

class DQNAgent:
    def __init__(self, action_size,model_path = None, state_size=(1, 13, 16, 4),render = True):
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
        if not model_path == None:
            self.model.load_weights(model_path)
        self.optimizer = Adam(self.learning_rate)

        # 타깃 모델 초기화
        self.update_target_model()
        self.avg_q_max, self.avg_loss = 0, 0

        self.writer = tf.summary.create_file_writer('summary/newEnvDQN2')
        self.model_path = os.path.join(os.getcwd(), 'save_model_newEnvDQN2', 'model')

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model(history)
            return np.argmax(q_value[0])

    def append_sample(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))

    def draw_tensorboard(self, score, step, episode):
        with self.writer.as_default():
            tf.summary.scalar('Total Reward/Episode', score, step=episode)
            tf.summary.scalar('Average Max Q/Episode',
                              self.avg_q_max / float(step), step=episode)
            tf.summary.scalar('Duration/Episode', step, step=episode)
            tf.summary.scalar('Average Loss/Episode',
                              self.avg_loss / float(step), step=episode)

    def train_model(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        batch = random.sample(self.memory, self.batch_size)

        history = np.array([sample[0][0] / 255. for sample in batch], dtype=np.float32)
        actions = np.array([sample[1] for sample in batch])
        rewards = np.array([sample[2] for sample in batch])
        next_history = np.array([sample[3][0] / 255. for sample in batch], dtype=np.float32)
        deads = np.array([sample[4] for sample in batch])

        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            predicts = self.model(history)
            one_hot_action = tf.one_hot(actions, self.action_size)
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)

            target_predicts = self.target_model(next_history)
            target_predicts = tf.stop_gradient(target_predicts) #학습도중 타깃모델학습되지 않도록

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
        model_path = None
        #model_path = './save_model_newEnvDQN/model'
        agent = DQNAgent(action_size=len(SIMPLE_MOVEMENT),
                         model_path = model_path,render = self.game_render)

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

                action = agent.get_action(history)
                next_state, reward, done, info = env.step(action)

                if self.pixel_render:
                    self.is_None = False
                    self.tiles = next_state
                    self.repaint()

                next_state = np.reshape([next_state], (1, 13, 16, 1))
                next_history = np.append(next_state, history[:, :, :, :3], axis=3)

                agent.avg_q_max += np.amax(agent.model(np.float32(history / 255.))[0])

                score += reward

                agent.append_sample(history, action, reward, next_history, info['dead'])

                if len(agent.memory) >= agent.train_start:
                    agent.train_model()
                    if global_step % agent.update_target_rate == 0:
                        agent.update_target_model()
                history = next_history
                if done:
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

            if e % 50 == 0:
                agent.model.save_weights("./save_model_newEnvDQN2/model", save_format="tf")

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

if __name__ == "__main__":
    App = QApplication(sys.argv)
    window = Window(game_render = True, pixel_render=True)
    sys.exit(App.exec())


