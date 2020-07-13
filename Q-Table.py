import gym
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Q_Table():

	def __init__(self, env, epsilon, epsilon_decay, alpha, gamma, n_bins, verbose=False):
		self.verbose = verbose
		self.epsilon = epsilon
		self.ep_decay = epsilon_decay
		self.alpha = alpha
		self.gamma = gamma
		self.state_log = []
		self.obsv_log = []
		self.observation = env.reset()
		self.n_actions = env.action_space.n
		self.obsv_shape = len(self.observation)
		self.n_bins = n_bins + 2
		self.state_records = {}
		self.n_states = 0
		self.policy = []
		self.state_vals = []
		self.frequency = []
		self.probabilities = []
		for i in range(self.n_actions):
			self.frequency.append([0])
			self.probabilities.append([0])
		self.terminal_states = []

		if verbose:
			print('Probability table len:',len(self.probabilities))
			print('Frequency table len:',len(self.frequency))
			print('Terminal table len:',len(self.terminal_states))

	def init_bins(self, spans):
		self.bins = []
		for span in spans:
			width = np.abs(span[0] - span[1]) / self.n_bins
			span = (span[0] - width, span[1] + width)
			self.bins.append(pd.cut(span, bins=self.n_bins, retbins=True)[1][1:-1])
			
	def log_info(self, observation, state):
		if len(self.obsv_log) > 100:
			del self.obsv_log[0]
		self.obsv_log.append(observation)
		if len(self.state_log) > 100:
			del self.state_log[0]
		self.state_log.append(state)

	def adapt_state_space(self):
		self.n_states = len(self.state_records)
		self.policy.append(np.random.randint(0,self.n_actions))
		self.terminal_states.append(0)
		self.state_vals.append(0)
		for a in range(self.n_actions):
			self.frequency[a].append(0)
			self.probabilities[a].append(0)
			self.update_probability(a, -1)

	def allocate_state(self, complex_s):
		if complex_s in self.state_records:
			sstate = self.state_records[complex_s]
		else:
			sstate = len(self.state_records)
			self.state_records[complex_s] = sstate
			self.adapt_state_space()
		return sstate

	def classify_discrete_obsv(self, observation):
		complex_state = 0
		for i,val in enumerate(observation):
			digitized = np.digitize(x=val, bins=self.bins[i]) * (self.n_bins ** i)
			if self.verbose:
				print('State attribute',i,'\n\tActual:',val,'Updated:', digitized)
			complex_state += digitized
		state = self.simplify_state(complex_state)
		self.log_info(observation, state)
		return state

	def solve_value_table(self):
		#initialize value table random
		self.epsilon *= self.ep_decay
		new_vals = np.copy(self.state_vals)
		for state in range(self.n_states):
			action = int(self.policy[state])
			s = np.argmax(self.probabilities[action])
			r = not self.terminal_states[s]
			self.state_vals[state] = self.probabilities[action][s] * (r + self.gamma * new_vals[s])
			if self.verbose:
				print(f'Probability of {action}->{s}:',self.probabilities[action][s])
				print(f'State {state}, value adjusted by {r + self.gamma * new_vals[s]}')

	def update_policy(self, state):
		if np.random.randint(1,11) <= self.epsilon:
			return np.random.randint(0,self.n_actions)
		Q_vals = np.zeros((self.n_actions))
		for a in range(self.n_actions):
			for s in range(self.n_states):
				Q_vals[a] += self.probabilities[a][s] * (self.state_vals[s] * self.gamma + 1)
		self.policy[state] = np.argmax(Q_vals)
		if self.verbose:
			print(f'State: {state}, policy adjusted to {self.policy[state]}')

	def update_probability(self, action, new_state):
		if new_state >= 0:
			self.frequency[action][new_state] += 1
		total = max(1,np.sum(self.frequency[action]))
		if self.verbose:
			print(f'Frequency of {action}: {total}')
		for new_state in range(self.n_states):
			self.probabilities[action][new_state] = self.frequency[action][new_state] / total

	def display_history(self, t_steps):
		# Plot Rewards
		obsv = np.array(self.obsv_log)
		state = np.array(self.state_log)
		fig, ax = plt.subplots(2,2)
		ax[0][0].plot(obsv[:,0::2], obsv[:,1::2])
		ax[0][0].set_xlabel('Observation Actual')
		ax[0][1].plot(state, np.arange(len(state)))
		ax[0][1].set_xlabel('Observation Discretized')
		ax[1][0].plot(np.arange(len(t_steps)), t_steps)
		ax[1][0].set_xlabel('T_Steps over T')
		ax[1][1].plot(np.arange(len(self.state_vals)), self.state_vals)
		ax[1][1].set_xlabel('Rewards vs States')
		plt.show()
		cv.waitKey()

		if self.verbose:
			print(self.terminal_states)


# TODO: check this link out
# Bellman implementation, doing it your way doesn't work!


def main():
	n_episodes = 100
	n_bins = 8
	epsilon = 7
	epsilon_decay = .9
	gamma = .2
	alpha = .9
	env = gym.make('CartPole-v0')
	t_steps = []

	spans = [[-2.4, 2.4],[-2, 2],[-1, 1],[-3.5, 3.5]]

	q_table = Q_Table(env, epsilon, epsilon_decay, gamma, alpha, n_bins, verbose=False)
	q_table.init_bins(spans)

	for i_episode in range(n_episodes):

		observation = env.reset() # update the observation and reset the environment
		state = q_table.classify_discrete_obsv(observation)
		q_table.solve_value_table()

		for t in range(200):
			if n_episodes - i_episode <= 4:
				q_table.epsilon = 0
				env.render()
			action = q_table.policy[state]
			observation, reward, done, info = env.step(action)

			new_state = q_table.classify_discrete_obsv(observation)
			q_table.update_probability(action, new_state)
			state = new_state
			
			if q_table.verbose:
				print('Policy:', q_table.policy)
				print(f'Probabilities from {action}', q_table.probabilities[action])

			if done:
				print(f"Episode {i_episode} finished after {t+1} timesteps")
				q_table.terminal_states[state] = True
				t_steps.append(t+1)
				break

	q_table.display_history(t_steps)
	env.close()


if __name__=="__main__":
	main()
