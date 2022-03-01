from setup_env import MyDoom
from time import sleep

_env = MyDoom(render=True)

total_steps = 100

state = _env.reset()
for step in range(total_steps):    
    state, reward, done, info = _env.step(_env.action_space.sample())
    print(info)
    sleep(0.2)
_env.close()