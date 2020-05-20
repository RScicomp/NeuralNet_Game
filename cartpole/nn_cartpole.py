# network:
# -------------------------------------------------------------------------------------
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

classifier = Sequential()
classifier.add(Dense(input_dim=4, output_dim=6, init='uniform', activation='sigmoid'))
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# -------------------------------------------------------------------------------------
# agent:
class Agent():
    def __init__(self,env):
        self.action_size = env.action_space.n

    def get_action(self, state):
        cart_pos = state[0]
        cart_vel = state[1]
        pole_ang = state[2]
        pole_vel = state[3]
        return 0


# -------------------------------------------------------------------------------------
# game:
import gym

env_name = "CartPole-v1"
env = gym.make(env_name)
state = env.reset()

agent = Agent(env)
for _ in range(200):
    action = agent.get_action(state)
    state, reward, done, info = env.step(action)
    env.render()
