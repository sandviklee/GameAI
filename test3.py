import flappy_bird_gym
from queue import Queue

env = flappy_bird_gym.make("FlappyBird-v0")

env.reset()
state = env.reset()
rbuf = []

def trainNetwork():


while True:
    prev_state = state
    action = env.action_space.sample()
    state , reward, done, info = env.step(action)
    
    rbuf.append((prev_state, state, action, reward))
    #env.render()
    
    if done:
        state = env.reset()

