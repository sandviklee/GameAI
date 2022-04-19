import time
import flappy_bird_gym
env = flappy_bird_gym.make("FlappyBird-v0")

obs = env.reset()
action = env.action_space.sample()

next_state, reward, done, _ = env.step(action)
print(env.obser)
