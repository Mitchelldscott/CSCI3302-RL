import gym
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Q_Table():

	def __init__(self, env, epsilon, alpha, gamma, n_bins, bins=[], verbose=False):
		self.verbose = verbose
		self.epsilon = epsilon
		self.alpha = alpha
		self.gamma = gamma
		self.state_log = []
		self.obsv_log = []
		self.observation = env.reset()
		self.n_actions = env.action_space.n
		self.obsv_shape = len(self.observation)
		self.n_states = n_bins ** self.obsv_shape
		self.table = np.random.uniform(low=0, high=1, size=(self.n_states, self.n_actions))

	def init_bins(self, spans, n_bins):
		self.bins = []
		self.n_bins = n_bins + 2
		self.terminal_bins = []

		for span in spans:
			width = np.abs(span[0] - span[1]) / n_bins
			span = (span[0] - width, span[1] + width)
			self.bins.append(pd.cut(span, bins=n_bins + 2, retbins=True)[1][1:-1])

	def classify_obsv(self, observation):
		state = 0
		if len(self.obsv_log) > 100:
			del self.obsv_log[0]
		self.obsv_log.append(observation)
		for i,val in enumerate(observation):
			encoded_bin = np.digitize(x=val, bins=self.bins[i]) * (self.n_bins ** i)
			if self.verbose:
				print('State attribute',i,'\n\tActual:',val,'Updated:', encoded_bin)
			state += encoded_bin
		if len(self.state_log) > 100:
			del self.state_log[0]
		self.state_log.append(state)
		return state
			

	def is_terminal(self, state):
		bin_vals = [0] * self.obsv_shape
		for i in range(self.obsv_shape - 1, 0, -1)
			oby = state % (self.n_bins)
			if oby > 0:
				if oby == self.n_bins - 1:
					return True
				state -= oby
			else:
				state =/ (self.n_bins)
				return True
		return False

	"""def discretize(self, observation):
		state = 0
		if len(self.obsv_log) > 100:
			del self.obsv_log[0]
		self.obsv_log.append(observation[1::2])
		for i,val in enumerate(observation[1::2]):
			encoded_bin = np.digitize(x=val, bins=self.bins[i]) * (len(self.bins) ** i)
			if self.verbose:
				print('State attribute',i,'\n\tActual:',val,'Updated:', encoded_bin)
			state += encoded_bin
		if len(self.state_log) > 100:
			del self.state_log[0]
		self.state_log.append(state)
		return state"""

"""	def apply_policy(self, state):
		q = self.table[state]
		if np.random.randint(0,10) < self.epsilon:
			maxq = max(q)
			action = np.where(q==maxq)[0][0]
		else:
			maxq = min(q)
			action = np.where(q==maxq)[0][0]
		if self.verbose:
			print('Discrete State:',state,'Action:',action)
		return action"""

"""	def update_table(self, state, action, reward):
		if self.verbose:
			print('Updating table at',state,action)
		self.table[state][action] += reward"""

	def get_state_value(self, state, depth=0, max_d=10):
		max_val = 0
		if depth > threashold:
			return 1
		if self.is_terminal(state):
			return 0 
		for a in range(self.n_actions):
			for s in range(self.n_states):
				val += self.probability[state][a][s] * self.get_state_value(depth + 1, s, max_d=threashold)
			val = (val * self.gamma) + 1
			if val > max_v:
				max_val = val
				max_a = a

		if depth == 0:
			return max_a

		return max_val

	def update_probability(state, action, new_state):
		self.frequency[state][action][new_state] += 1
		total = np.sum(self.frequency[state][action])

		for new_state in range(self.n_states):
			self.probability[state][action][new_state] = self.frequency[state][action][new_state] / total

	def display_history(self, t_steps):
		# Plot Rewards
		obsv = np.array(self.obsv_log)
		state = np.array(self.state_log)
		fig, ax = plt.subplots(1,2)
		ax[0].plot(obsv[:,0::2], obsv[:,1::2])
		ax[0].set_xlabel('Observation Actual')
		ax[1].plot(np.arange(len(state)), state)
		ax[1].set_xlabel('Observation Discretized')
		plt.show()
		cv.waitKey()

		print('Table Shape:', self.table.shape)
		if self.verbose:
			print(self.table)




# TODO: check this link out
# Bellman implementation, doing it your way doesn't work!


def main():
	n_episodes = 200
	n_bins = 4
	epsilon = 8
	gamma = .5
	alpha = .9
	env = gym.make('CartPole-v0')
	t_steps = []

	bins = [[-2.4, 2.4],[-2, 2],[-1, 1],[-3.5, 3.5]]

	q_table = Q_Table(env, epsilon, gamma, alpha, n_bins, bins=bins, verbose=False)

	for i_episode in range(n_episodes):

		observation = env.reset() # update the observation and reset the environment
		reward_sum = 0
		#q_table.epsilon = ((n_episodes - i_episode) / n_episodes ** 2) * 10

		for t in range(200):
			if n_episodes - i_episode <= 1:
				q_table.epsilon = 0
				env.render()

			state = q_table.discretize(observation)
			action = q_table.apply_policy(state)
			observation, reward, done, info = env.step(env.action_space.sample())
			reward_sum += reward
			reward = reward * gamma + reward_sum

			q_table.update_table(state, action, reward)
			#q_table.display_history()

			if done:
				print(f"Episode {i_episode} finished after {t+1} timesteps")
				t_steps.append(t+1)
				break

	q_table.display_history(t_steps)
	env.close()


if __name__=="__main__":
	main()
