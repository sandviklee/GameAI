
import time
import flappy_bird_gym
from collections import deque
import torch
import random

import torch.nn as nn

torch.set_default_tensor_type("torch.DoubleTensor")

env = flappy_bird_gym.make("FlappyBird-v0")
env.reset()
state = env.reset()
rbuf = deque(maxlen=1000)
gamma = 0.98
epsilon_initial = 0.0027353
epsilon_min = 0.0001
epsilon_step_size = 0.00001

def update_epsilon(epsilon):
    return(max(epsilon_min, min(epsilon_initial, epsilon - epsilon_step_size)))



class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )

    def forward(self, x):
        x = torch.tensor(x).double()

        logits = self.linear_relu_stack(x)
        return logits
    



qnet = QNetwork()

optimizer = torch.optim.Adam(qnet.parameters(),lr=0.01)
criterion = nn.MSELoss()

def train_network():
    minibatch = random.sample(rbuf,16)
    for prev_state, state, action, reward in minibatch:
        output = qnet(state)
        prediction = qnet(prev_state)
        target = reward + gamma*torch.max(output)
        target_f = qnet(prev_state)
        target_f[action] = target
        loss = criterion(prediction, target_f)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

i = 0
epsilon = epsilon_initial
done = False

while (i<10000):
    
    state = env.reset()
    i+=1
    done = False
    epsilon = update_epsilon(epsilon)

    while not done:
        prev_state = state
        action = qnet(state).max(0)[1].numpy() if random.uniform(0,1)>epsilon else env.action_space.sample()
        
        state , reward, done, info = env.step(action)
        
        
        
        if (abs(state[1])>0.1):
            reward -= 10
        reward +=5
        
        rbuf.append((prev_state, state, action, reward))
        
        if len(rbuf)>16 and i%100==0:
            train_network()
        if i%100==0:
            env.render()
            print(state)
            time.sleep(1/60) 

        
        if done:
            state = env.reset()

        
    
