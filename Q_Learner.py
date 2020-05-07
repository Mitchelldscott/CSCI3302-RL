import gym
import random
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class QLearner():

	def __init__(self, env, epsilon, epsilon_decay, gamma, alpha, n_bins=-1, bins=[], n_states=0):
		obsv = env.reset()
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.gamma = gamma
		self.alpha = alpha
		self.n_states = n_states
		self.n_actions = env.action_space.n
		if n_bins >= 0:
			self.n_bins = n_bins
			self.init_bins(bins) # helps to classify discrete spans
		self.value_table = np.random.uniform(low=0, high=1, size=(self.n_states,))
		self.Q_table = np.random.uniform(low=0, high=1, size=(self.n_states, self.n_actions))
		self.policy = np.random.uniform(low=0, high=2, size=(self.n_states,)).astype(np.int)
		self.p_table = np.zeros((self.n_states, self.n_actions, self.n_states))
		self.freq_table = np.zeros((self.n_states, self.n_actions, self.n_states))
		self.state_zips = {} # Classifies complex states
		self.terminal_states = np.zeros((self.n_states,)) # SHAPE = (n_states) track which states are OOB

	def init_bins(self, spans):
		self.bins = []
		for span in spans:
			width = np.abs(span[0] - span[1]) / self.n_bins
			span = (span[0] - width, span[1] + width)
			self.bins.append(pd.cut(span, bins=self.n_bins, retbins=True)[1][1:-1])
			
	def adjust_state_space(self):
		self.n_states = len(self.state_zips)
		new_t_states = np.zeros((self.n_states,)).astype(np.int)
		new_t_states[:-1] = self.terminal_states
		self.terminal_states = new_t_states
		v_table = np.zeros((self.n_states,))
		v_table[:-1] = self.value_table
		self.value_table = v_table
		q_table = np.random.uniform(low=0, high=1, size=(self.n_states, self.n_actions))
		q_table[:-1] = self.Q_table
		self.Q_table = q_table
		self.policy = np.random.uniform(low=0, high=2, size=(self.n_states,)).astype(np.int)
		new_p_table = np.zeros((self.n_states,self.n_actions,self.n_states))
		new_fr_table = np.random.uniform(low=0, high=1, size=(self.n_states,self.n_actions,self.n_states)).astype(np.int)
		for s in range(self.n_states - 1):
			for a in range(self.n_actions):
				new_p_table[s][a][:-1] = self.p_table[s][a]
				new_fr_table[s][a][:-1] = self.freq_table[s][a]
		self.p_table = new_p_table
		self.freq_table = new_fr_table
			
	def allocate_state(self, complex_s):
		if complex_s in self.state_zips:
			sstate = self.state_zips[complex_s]
		else:
			sstate = len(self.state_zips)
			self.state_zips[complex_s] = sstate
			self.adjust_state_space()
		return sstate

	def zip_observation(self, observation):
		complex_state = 0
		for i,val in enumerate(observation):
			digitized = np.digitize(x=val, bins=self.bins[i]) * (self.n_bins ** i)
			complex_state += digitized
		state = self.allocate_state(complex_state)
		return state

	def simple_value_iteration(self, max_depth=30, threashold=.00000000001):
		if self.epsilon > np.random.rand():
			return np.random.randint(0,self.n_actions)
		for i in range(max_depth):
			orig_table = np.copy(self.value_table)
			for s in range(1,self.n_states):
				max_q = 0
				for a in range(self.n_actions):
					Q_val = 0
					for next_s in range(1,self.n_states):
						reward = not self.terminal_states[next_s]
						Q_val += self.p_table[s][a][next_s] * (self.gamma * orig_table[next_s] + reward)
					if max_q < Q_val:
						max_q = Q_val
						self.policy[s] = a
				if max_q == 0:
					max_q = np.random.uniform(low=0, high=1)
				self.value_table[s] = max_q
			if np.fabs(np.sum(orig_table) - np.sum(self.value_table)) < threashold:
				break

	def update_probabilities(self):
		for s in range(self.n_states):
			for a in range(self.n_actions):
				total = np.sum(self.freq_table[s][a])
				if total > 0:
					for next_s in range(self.n_states):
						self.p_table[s][a][next_s] = self.freq_table[s][a][next_s] / total

	def update_q_table(self, state, action, new_state, reward, done):
		if done:
			target = -5
		else:
			target = reward + self.gamma * np.max(self.Q_table[new_state])
		self.Q_table[state][action] += self.alpha * (target - self.Q_table[state][action])

	def track_event(self, state, action, new_state, reward, done=0):
		#self.freq_table[state][action][new_state] += 1		# for value iteration
		#self.update_probabilities()		# for value iteration
		self.epsilon *= self.epsilon_decay
		if done:
			self.terminal_states[new_state] = 1
		self.update_q_table(state, action, new_state, reward, done)

	def display_history(self, t_steps, test_steps, value_table):
		# Plot Info
		print('Total Average Score: ', np.sum(t_steps) / len(t_steps))
		print('Average Test Score: ', np.sum(test_steps) / len(test_steps))
		print('Terminal States:', self.terminal_states)
		print('Policy: ', self.policy)
		#print('Value Table: ', value_table)
		print('Q Table: ', self.Q_table)
		fig = plt.figure()
		ax0 = fig.add_subplot(121)
		ax1 = fig.add_subplot(122, projection='3d')
		ax0.plot(np.arange(len(t_steps)), t_steps)
		ax0.set_xlabel('T_Steps over Time')
		x = []
		y= []
		z = []
		for s in range(self.n_states):
			for a in range(self.n_actions):
				x.append(s)
				y.append(a)
				z.append(self.Q_table[s][a])
		ax1.scatter(x,y,z)
		ax1.set_xlabel('Q Values')
		plt.show()
		cv.waitKey()




