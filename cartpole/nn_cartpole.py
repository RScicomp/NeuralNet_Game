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
        self.x_train = np.array([])
        self.y_train = np.array([])

    def get_action(self, state):
        cart_pos = state[0]/10
        cart_vel = state[1]/100
        pole_ang = state[2]/100
        pole_vel = state[3]/100

        inputs = np.array([cart_pos, cart_vel, pole_ang, pole_vel])
        inputs = inputs.flatten()
        print(inputs.shape)
        if cart_pos > 0:
            self.x_train = np.append(self.x_train, inputs)
            self.y_train = np.append(self.y_train, [0])
        elif cart_pos < 0:
            self.x_train = np.append(self.x_train, inputs)
            self.y_train = np.append(self.y_train, [1])

        return classifier.predict_classes(np.array(state))

    def train(self):
        classifier.fit(self.x_train,self.y_train)

# -------------------------------------------------------------------------------------
# game:
import gym

env_name = "CartPole-v1"
env = gym.make(env_name)
state = env.reset()
reward = 0

agent = Agent(env)

def random_games():
    for episode in range(10):
        env.reset()
        for _ in range(200):
            #action = agent.get_action(state)
            env.render()
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            if done:
                break

random_games()