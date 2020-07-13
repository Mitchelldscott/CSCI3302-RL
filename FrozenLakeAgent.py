import gym
import cv2 as cv
import numpy as np
import pandas as pd
from Q_Learner import QLearner
import matplotlib.pyplot as plt



def main():
	gamma = .5
	alpha = .4
	epsilon = 5
	t_steps = []
	n_episodes = 1
	epsilon_decay = .9
	env = gym.make('FrozenLake-v0')
	learner = QLearner(env, epsilon, epsilon_decay, gamma, alpha, n_states=env.observation_space.n)

	opt = 0
	for i_episode in range(n_episodes):

		observation = env.reset() # update the observation and reset the environment
		prev_obsv = 0
		#state = learner.zip_observation(observation) # Discretize the observation
		for t in range(200):
			if n_episodes - i_episode <= 20:
				learner.epsilon = 0
				env.render()
				action = np.argmax(learner.Q_table[observation])
				prev_obsv = observation
				observation, reward, done, info = env.step(action)
			else:
				#learner.simple_value_iteration()
				if np.random.randint(1,10) < learner.epsilon:
					action = env.action_space.sample()
					#learner.epsilon = max(2, learner.epsilon * epsilon_decay)
				else:
					action = np.argmax(learner.Q_table[observation])#learner.policy[state]
				observation, reward, done, info = env.step(action)
				#new_state = learner.zip_observation(observation)
				learner.track_event(prev_obsv, action, observation, reward, done)
			print(reward)

			if done:
				print(f"Episode {i_episode} finished after {t+1} timesteps")
				t_steps.append(t+1)
				opt = t + 1
				break

	learner.display_history(t_steps, learner.value_table)
	env.close()


if __name__=="__main__":
	main()