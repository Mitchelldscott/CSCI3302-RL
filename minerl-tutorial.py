import minerl
import gym
import logging

def main():
	env = gym.make('MountainCar')

	while not done:
		action = 2
		
		obs, reward, done, _ = env.step(action)
		
if __name__=="__main__":
	main()