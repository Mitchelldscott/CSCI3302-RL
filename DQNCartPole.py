import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DQNAgent:

	def __init__(self, state_size, action_size):
		self.n_states = state_size
		self.n_actions = action_size
		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_decay = 0.995
		self.epsilon_min = .01
		self.lr_decay = 
		self.learning_rate = 0.01
		self.model = self.build_model()
		self.memory = deque(maxlen=2000)

	def build_model(self):
		model = Sequential()
		model.add(Dense(24, input_dim=self.n_states, activation='relu'))
		model.add(Dense(24, activation='relu'))
		model.add(Dense(self.n_actions, activation='linear'))
		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
		model.summary()
		return model

	def record(self, state, action, reward, new_state, done):
		self.memory.append((state, action, reward, new_state, done))

	def act(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.n_actions)
		act_values = self.model.predict(state)
		return np.argmax(act_values[0])

	def track_event(self, batch_size):
		mini_batch = random.sample(self.memory, batch_size) 
		for state, action, reward, next_state, done in mini_batch:
			target = reward
			if not done:
				target = (reward + self.gamma * np.max(self.model.predict(next_state)[0]))
			target_y = self.model.predict(state)
			target_y[0][action] = target
			self.model.fit(state, target_y, epochs=1, verbose=0)
		if self.epsilon >= self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def load(self, filename):
		self.model.load_weights(filename)

	def save(self, fileName):
		self.model.save(filename)


def run(env, n_states, n_actions, batch_size, n_episodes):
	max_t = 200
	agent = DQNAgent(state_size, action_size)
	time_steps = []
	for i in range(n_episodes):
		state = np.reshape(env.reset(), [1, n_states])
		for t in range(max_t):
			if n_episodes - i <= 5:
				env.render()
			action = agent.act(state)
			new_state, reward, done, _= env.step(action)
			if done:
				reward = -10
			new_state = np.reshape(new_state, [1, n_states])
			agent.record(state, action, reward * (t/max_t), new_state, done)
			if done:
				print(f'Episode {i} finished after {t} steps, e {agent.epsilon}')
				time_steps.append(t)
				break
		agent.track_event(np.min([batch_size, np.sum(time_steps)]))
	print(f'Average time {np.mean(time_steps)}')


if __name__=='__main__':
	env = gym.make('CartPole-v0')
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n
	batch_size = 64
	n_episodes = 1001
	run(env, state_size, action_size, batch_size, n_episodes)



