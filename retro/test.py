import retro 
import time
import numpy as np
#print(retro.data.list_games())

env = retro.make("StreetFighterIISpecialChampionEdition-Genesis")
# python -m retro.import . # Run this from the roms folder, or where you have your game roms 

print(env.observation_space.shape)
print(env.action_space.shape)


# Reset game to starting state
obs = env.reset()
# Set flag to flase
done = False
for game in range(1): 
    while not done: 
        if done: 
            obs = env.reset()
        env.render()
        #obs, reward, done, info = env.step(env.action_space.sample())
        action = np.zeros(12)
        action[5] = 1  
        #print(action)
        obs, reward, done, info = env.step(action)
        time.sleep(0.01)
        print(reward)