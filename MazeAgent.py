import random
import cv2 as cv
import numpy as np
import pandas as pd
from collections import deque
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from mpl_toolkits.mplot3d import Axes3D
from keras.models import Sequential, load_model

class QLearner():

	def __init__(self, epsilon, epsilon_decay, epsilon_min, gamma, alpha, n_bins=-1, bins=[], n_states=0, n_actions=0):
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.epsilon_min = epsilon_min
		self.gamma = gamma
		self.alpha = alpha
		self.n_states = n_states
		self.n_actions = n_actions
		if n_bins >= 0:
			self.n_bins = n_bins
			self.init_bins(bins) # helps to classify discrete spans
		self.state_zips = {} # Classifies complex states
		self.history = np.zeros((self.n_states, self.n_actions))
		self.Q_table = np.random.uniform(low=0, high=1, size=(self.n_states, self.n_actions))

	def init_bins(self, spans):
		self.bins = []
		for span in spans:
			width = np.abs(span[0] - span[1]) / self.n_bins
			span = (span[0] - width, span[1] + width)
			self.bins.append(pd.cut(span, bins=self.n_bins, retbins=True)[1][1:-1])

	def adjust_state_space(self):
		self.n_states = len(self.state_zips)
		q_table = np.random.uniform(low=0, high=1, size=(self.n_states, self.n_actions))
		q_table[:-1] = self.Q_table
		self.Q_table = q_table
		hist = np.zeros((self.n_states, self.n_actions))
		hist[:-1] = self.history
		self.history = hist

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

	def get_best_action(self, state, mask):
		m = np.min(self.Q_table[state])
		action = mask[0]
		for a in mask:
			if self.Q_table[state][a] >= m:
				action = a
				m = self.Q_table[state][a]
		return action

	def epsilon_greedy(self, state, mask):
		if np.random.rand() <= self.epsilon:
			action = mask[np.random.randint(len(mask))]
		else:
			action = self.get_best_action(state, mask)
		return action

	def frequency(self, state, mask, threashold):
		val = np.max(self.history[state])
		for a in mask:
			if self.history[state][a] <= val:
				action = a
				val = self.history[state][a]
		if val > threashold:
			return self.get_best_action(state, mask)
		return action

	def relative_frequency(self, state, mask, threashold0, threashold1):
		total_f = np.sum(self.history[state])
		if total_f > threashold1:
			return self.get_best_action(state, mask)
		for i, a in enumerate(self.history[state]):
			if a / total_f <= threashold0:
				return i
		return self.get_best_action(state, mask)

	def update_q_table(self, state, action, new_state, reward, done):
		if done:
			target = 1
		else:
			target = reward + self.gamma * np.max(self.Q_table[new_state])
		self.Q_table[state][action] += self.alpha * (target - self.Q_table[state][action])

	def track_event(self, state, action, new_state, reward, done=0):
		self.history[state][action] += 1
		self.epsilon = max(self.epsilon_decay * self.epsilon, epsilon_min)
		if done:
			self.terminal_states[new_state] = 1
		self.update_q_table(state, action, new_state, reward, done)

	def display_history(self, t_steps, rewards):
		# Plot Info
		print('Total Average Time: ', np.sum(t_steps) / len(t_steps))
		print('Average Score: ', np.sum(rewards) / len(rewards))
		print('Frequencies: ', self.history)
		print('Q-Table', self.Q_table)
		fig = plt.figure()
		ax0 = fig.add_subplot(131)
		ax1 = fig.add_subplot(132)
		ax2 = fig.add_subplot(133, projection='3d')
		ax0.plot(np.arange(len(t_steps)), t_steps)
		ax0.set_xlabel('Steps over Time')
		ax1.plot(np.arange(len(rewards)), rewards)
		ax1.set_xlabel('Rewards over Time')
		x = []
		y= []
		z = []
		for s in range(self.n_states):
			for a in range(self.n_actions):
				x.append(s)
				y.append(a)
				z.append(self.Q_table[s][a])
		ax2.scatter(x,y,z)
		ax2.set_xlabel('States')
		ax2.set_ylabel('Actions')
		plt.show()
		cv.waitKey()

class Maze_Env:
	def __init__(self, columns=5, rows=5, spawns=[(0,0)], goals=[(4,4)], obstacles=[], random_obstacles=0):
		self.n_actions = 4
		self.agent_pose = spawns[0]
		self.spawns = spawns
		self.curr_spawn = 0
		self.goals = {}
		for goal in goals: # This is a key update to the algorithm. Once a target is found finding it again is useless
			self.goals[goal] = 1
		self.obstacles = obstacles
		if random_obstacles:
			for i in range(int(columns * rows / random_obstacles)):
				point = (np.random.randint(0,columns), np.random.randint(0,rows))
				if not point in spawns and not point in goals:
					self.obstacles.append(point)
		self.size = (columns, rows)

	def reset(self, targets):
		self.curr_spawn = (self.curr_spawn + 1) % len(self.spawns)
		self.agent_pose = self.spawns[self.curr_spawn]
		for goal in self.goals:
			self.goals[goal] = 1  
		return self.agent_pose

	def step(self, action):
		pose = self.agent_pose
		if action == 0 and pose[0] + 1 < self.size[0]:
			self.agent_pose = (pose[0] + 1, pose[1])
		elif action == 1 and pose[0] - 1 >= 0:
			self.agent_pose = (pose[0] - 1, pose[1])
		elif action == 2 and pose[1] + 1 < self.size[1]:
			self.agent_pose = (pose[0], pose[1] + 1)
		elif action == 3 and pose[1] - 1 >= 0:
			self.agent_pose = (pose[0], pose[1] - 1)
		if self.agent_pose in self.obstacles:
			return self.agent_pose, -5, 0
		done = 1
		found = 0
		for goal in self.goals:
			if self.agent_pose == goal:
				found = self.goals[goal]
				self.goals[goal] = 0
			if self.goals[goal]:
				done = 0
		return self.agent_pose, found, done

		# This function simulates rambo deciding what actions are possible.
	def get_action_mask(self):
		pose = self.agent_pose
		mask = []
		if pose[0] + 1 < self.size[0]:
			mask.append(0)
		if pose[0] - 1 >= 0:
			mask.append(1)
		if pose[1] + 1 < self.size[1]:
			mask.append(2)
		if pose[1] - 1 >= 0:
			mask.append(3)
		return mask

	def render(self):
		mark_size = (2000,2000)
		plt.figure(figsize=(8, 8))
		x = []
		y = []
		for pose in self.obstacles:
			c,r = pose
			x.append(c)
			y.append(r)
		plt.scatter(x, y, marker='s', c='black', s=mark_size)
		for goal in self.goals:
			plt.scatter(goal[0], goal[1], marker='s', c='r', s=mark_size)
		for spawn in self.spawns:
			plt.scatter(spawn[0], spawn[1], marker='s', c='g', s=mark_size)
		plt.scatter(self.agent_pose[0], self.agent_pose[1], marker='s', c='b', s=mark_size)
		plt.show()

	def dist_from_start(self):
		x = (self.spawns[self.curr_spawn][0] - self.agent_pose[0])**2
		y = (self.spawns[self.curr_spawn][1] - self.agent_pose[1])**2
		return np.sqrt(x + y)

	def dist_from_goals(self):
		dist = 0
		for goal in self.goals:
			x = (goal[0] - self.agent_pose[0])**2
			y = (goal[1] - self.agent_pose[1])**2
			dist += np.sqrt(x + y)
		return dist / len(self.goals)


cols = 5
rows = 5
end_points = [(0,4), (4,0), (4,4)]
obstacles = [(2,3), (3,3), (0,2), (2,2), (2,0), (1,3)]
maze = Maze_Env(columns=cols, rows=rows, obstacles=obstacles, goals=end_points)
gamma = .9
alpha = .4
epsilon = 0.9
epsilon_min = 0.01
epsilon_decay = .9
n_episodes = 2000
learner = QLearner(epsilon, epsilon_decay, epsilon_min, gamma, alpha, n_actions=4)

time_steps = []
rewards = []
agent_history = deque(maxlen=4)
done = False
for i in range(n_episodes):
	prev = (0,0)
	pose = maze.reset(end_points)
	p = (pose[0], pose[1], prev[0], prev[1])
	state = learner.allocate_state(p)
	reward = 0
	t = 0
	while 1:
		mask = maze.get_action_mask()
		action = learner.epsilon_greedy(state, mask)
		if n_episodes - i < 2:
			maze.render()
		pose, target, done = maze.step(action)
		p = (pose[0], pose[1], prev[0], prev[1])
		prev = pose
		new_state = learner.allocate_state(p)
		reward += (target * 5) - (0.1 * t)
		learner.track_event(state, action, new_state, reward)
		state = new_state
		if done:
			print(f'Targets Aquired in {t+1} steps')
			break
		t += 1
	rewards.append(reward)
	time_steps.append(t+1)
	print(f'Episode {i} Terminated after {t+1} steps, and scored {reward}')
learner.display_history(time_steps, rewards)

