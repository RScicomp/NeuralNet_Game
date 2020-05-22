# network:
# -------------------------------------------------------------------------------------
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

classifier = Sequential()
# 3 hidden layers
# flattened input layer to deal with shape problems
classifier.add(Flatten(input_shape=(4, 1)))
classifier.add(Dense(128, init='uniform', activation='relu'))
classifier.add(Dense(128, init='uniform', activation='relu'))
classifier.add(Dense(128, init='uniform', activation='relu'))
# output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# -------------------------------------------------------------------------------------
# agent:
# agent uses the neural net to train and predict
# TODO: make the neural net part of the agent for multiple versions of agents
class Agent():
    def __init__(self, env):
        self.action_size = env.action_space.n
        self.x_train = np.array([])
        self.y_train = np.array([])

    def get_action(self, state):
        state = np.array([state])
        state = state.reshape(1,4,1)
        return classifier.predict_classes(state)[0][0]

    def train(self, data):
        x = np.array([i[0] for i in data]).reshape(-1, len(data[0][0]), 1)
        y = np.array([i[1] for i in data])
        classifier.fit(x, y, epochs=10, verbose=1)
# -------------------------------------------------------------------------------------
# game environment:
import gym

env_name = "CartPole-v1"
env = gym.make(env_name)
state = env.reset()
# define agent
agent = Agent(env)


# this is the random agent, run this before hand to see the random player
def random_games():
    for episode in range(10):
        env.reset()
        for _ in range(500):
            env.render()
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            if done:
                break


# uses the gen_one bot
def gen_one_test():
    global state
    for episode in range(10):
        env.reset()
        for _ in range(500):
            env.render()
            action = agent.get_action(state)
            state, reward, done, info = env.step(action)
            if done:
                break


# uses the good (>= 50 points) games from random games to train
def gen_one():
    score = np.array([])
    data = []
    # running 500 random games
    for episode in range(500):
        env.reset()

        step_reward = 0
        game_data = []
        prev_obs = []
        # 500 steps in each game
        for _ in range(500):
            # save game data
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)

            if len(prev_obs) > 0:
                game_data.append([prev_obs, action])
            prev_obs = state
            step_reward += reward
            if done:
                break
        # filter good games from bad ones
        if step_reward >= 50:
            score = np.append(score, step_reward)
            for d in game_data:
                data.append(d)

    return data

# get data from good random games
gen_one_data = gen_one()
# train agent with the data
agent.train(gen_one_data)
# testing gen_one agent
gen_one_test()
