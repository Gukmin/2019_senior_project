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
    def __init__(self, action_size, state_size = (1, 13, 16 ,4), model_path,render = True):
        self.render = render

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        self.epsilon = 0.02

        # 모델과 타깃 모델 생성
        self.model = DQN(action_size, state_size)
        self.model.load_weights(model_path)

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model(history)
            return np.argmax(q_value[0])

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
        model_path = './save_model_newEnvDQN'
        agent = DQNAgent(action_size=len(SIMPLE_MOVEMENT),render = self.game_render)
        num_episode = 10
        for e in range(num_episode):
            done = False
            score = 0
            state = env.reset()

            history = np.stack([state,state,state,state],axis=2)
            history = np.reshape([history],(1,13,16,4))
            while not done:
                if self.game_render:
                    env.render()
                action = agent.get_action(history)
                next_state, reward, done, info = env.step(action)

                if self.pixel_render:
                    self.is_None = False
                    self.update()
                    self.tiles = next_state

                next_state = np.reshape([next_state],(1,13,16,1))
                next_history = np.append(next_state, history[:, :, :, :3], axis=3)

                score += reward

                if done:
                    print("episode: {:3d} | score : {:4.1f}".format(e, score))

    def InitWindow(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)
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
                painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))
                x_start = 5 + (10 * col)
                y_start = 5 + (10 * row)
                if self.tiles[row][col] == 85:
                    painter.setBrush(QBrush(Qt.red, Qt.SolidPattern))
                elif self.tiles[row][col] == 190:
                    painter.setBrush(QBrush(Qt.blue, Qt.SolidPattern))
                elif self.tiles[row][col] == 255:
                    painter.setBrush(QBrush(Qt.green, Qt.SolidPattern))
                painter.drawRect(x_start, y_start, 10, 10)

# main
if __name__ == "__main__":
    App = QApplication(sys.argv)
    window = Window(game_render=True, pixel_render=True)
    sys.exit(App.exec())
