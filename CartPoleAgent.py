import gym
import random
import cv2 as cv
import numpy as np
import pandas as pd
from Q_Learner import QLearner
import matplotlib.pyplot as plt



def main():
	n_bins = 4
	gamma = .9
	alpha = .5
	epsilon = .2
	t_steps = []
	done = False
	test_steps = []
	n_episodes = 1000
	epsilon_decay = .999
	env = gym.make('CartPole-v0')
	spans = [[-2.4, 2.4],[-2, 2],[-1, 1],[-3.5, 3.5]]
	learner = QLearner(env, epsilon, epsilon_decay, gamma, alpha, n_bins=n_bins, bins=spans)

	for i_episode in range(n_episodes):

		observation = env.reset() # update the observation and reset the environment
		state = learner.zip_observation(observation) # Discretize the observation

		for t in range(200):

			if n_episodes - i_episode <= 2:
				env.render()
				action = np.argmax(learner.Q_table[state])
				observation, reward, done, info = env.step(action)
				new_state = learner.zip_observation(observation)
				state = new_state
				
			else:
				if np.random.rand() <= learner.epsilon:
					action = env.action_space.sample()
				else:
					action = np.argmax(learner.Q_table[state])
				observation, reward, done, info = env.step(action)
				new_state = learner.zip_observation(observation)
				learner.track_event(state, action, new_state, t + 1, done)
				state = new_state

			if done:
				print(f"Episode {i_episode} finished after {t+1} timesteps")
				t_steps.append(t + 1)
				if n_episodes - i_episode <= 20:
					test_steps.append(t + 1)
				break

	learner.display_history(t_steps, test_steps, learner.value_table)
	env.close()


if __name__=="__main__":
	main()