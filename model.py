import random
import tensorflow as tf 
import pandas as pd 
import numpy as np 
from collections import deque

class AITRADE(object):
    def __init__(self, state_size, action_space=3, model_name="AITRADE"):
        super(AITRADE, self).__init__()
        self.state_size = state_size
        self.action_space = action_space
        self.model_name = model_name
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995
        self.model = self.model_builder()
    

    def model_builder(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=32, activation='relu', input_dim=self.state_size))
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        model.add(tf.keras.layers.Dense(units=self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=0.001))

        return model

    def trade(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)

        actions = self.model.predict(state)

        return np.argmax(actions[0])

    def batch_train(self, batch_size):
        batch = []
        for i in range(len(self.memory) - batch_size + 1, len(self.memory)):
            batch.append(self.memory[i])

        for state,action,reward,next_state,done in batch:
            if not done:
                reward = reward + self.gamma * np.max(self.model.predict(next_state)[0])
                target = self.model.predict(state)
                
                target[0][action] = reward
                self.model.fit(state, target, epochs=1, verbose=0)

            if self.epsilon > self.epsilon_final:
                self.epsilon *= self.epsilon_decay
                
                
            
        
        