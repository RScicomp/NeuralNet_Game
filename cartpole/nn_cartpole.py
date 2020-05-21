# network:
# -------------------------------------------------------------------------------------
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

classifier = Sequential()
# 3 hidden layers
classifier.add(Flatten(input_shape=(4, 1)))
classifier.add(Dense(128, init='uniform', activation='relu'))
classifier.add(Dense(128, init='uniform', activation='relu'))
classifier.add(Dense(128, init='uniform', activation='relu'))
# output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# -------------------------------------------------------------------------------------
# agent:
class Agent():
    def __init__(self, env):
        self.action_size = env.action_space.n
        self.x_train = np.array([])
        self.y_train = np.array([])

    def get_action(self, state):
        inputs = np.array()
        inputs = inputs.flatten()
        classifier.predict_classes(np.array(state))

    def train(self, data):
        x = np.array([i[0] for i in data]).reshape(-1, len(data[0][0]), 1)
        y = np.array([i[1] for i in data])
        classifier.fit(x, y, epochs=10, verbose=1)


# -------------------------------------------------------------------------------------
# game:
import gym


env_name = "CartPole-v1"
env = gym.make(env_name)
state = env.reset()

agent = Agent(env)


def random_games():
    for episode in range(1):
        env.reset()
        for _ in range(500):
            #action = agent.get_action(state)
            env.render()
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            if done:
                break


def gen_one():
    score = np.array([])
    data = []

    for episode in range(500):
        env.reset()

        step_reward = 0
        game_data = []
        prev_obs = []

        for _ in range(500):
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)

            if len(prev_obs) > 0:
                game_data.append([prev_obs, action])
            prev_obs = state
            step_reward += reward
            if done:
                break

        if step_reward >= 50:
            score = np.append(score, step_reward)
            for d in game_data:
                data.append(d)

    print(score)

    return data


gen_one_data = gen_one()

agent.train(gen_one_data)

#print(gen_one_data.shape)

