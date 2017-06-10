import os
import gym
import random
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
MSELoss_fn = torch.nn.MSELoss()
transform = T.Compose([T.ToPILImage(),
                       T.Scale(64, interpolation=Image.CUBIC),
                       T.ToTensor()])

# Utility functions
def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(torch.from_numpy(ndarray),
                    volatile=volatile, requires_grad=requires_grad).type(dtype)

def OU_noise(x, mu, epsilon):
    # Ornstein-Uhlenbeck process
    # dx_t = \theta*(\mu - x_t)dt + \sigma*dW_t
    # Here set \theta = 0.15, \sigma = 0.2
    return epsilon*(0.15*(mu - x) + 0.2*np.random.randn(1))

def fanin_init(size):
    if len(size) == 4:
        fanin = size[1]*size[2]*size[3]
    elif len(size) == 2:
        fanin = size[0]
    else:
        raise ('Invalid size shape')
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

def get_screen(env):
    screen_sz = 500
    bound = 150
    slice_range = slice(screen_sz / 2 - bound,
                        screen_sz / 2 + bound)
    screen = env.render(mode='rgb_array')
    view = screen[slice_range, slice_range, :].transpose(2,0,1)
    view = torch.from_numpy(np.ascontiguousarray(view))
    view = transform(view).unsqueeze(0) # NCHW
    return view


# Basic classes
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(state_dim[0], 32, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(1568, 200)
        self.bn4 = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, 200)
        self.bn5 = nn.BatchNorm1d(200)
        self.fc3 = nn.Linear(200, action_dim)
        self.init_weights()

    def init_weights(self, small_init=3e-4):
        self.conv1.weight.data = fanin_init(self.conv1.weight.data.size())
        self.conv2.weight.data = fanin_init(self.conv2.weight.data.size())
        self.conv3.weight.data = fanin_init(self.conv3.weight.data.size())
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-small_init, small_init)

    def forward(self, s):
        x = F.relu(self.bn1(self.conv1(s)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.fc1(x.view(x.size(0), -1))))
        x = F.relu(self.bn5(self.fc2(x)))
        x = F.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(state_dim[0], 32, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(1568, 200)
        self.bn4 = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, 200)
        self.bn5 = nn.BatchNorm1d(200)
        self.fc3 = nn.Linear(action_dim, 50)
        self.bn6 = nn.BatchNorm1d(50)
        self.fc4 = nn.Linear(250, 1)
        self.init_weights()

    def init_weights(self, small_init=3e-4):
        self.conv1.weight.data = fanin_init(self.conv1.weight.data.size())
        self.conv2.weight.data = fanin_init(self.conv2.weight.data.size())
        self.conv3.weight.data = fanin_init(self.conv3.weight.data.size())
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.fc4.weight.data.uniform_(-small_init, small_init)

    def forward(self, s, a):
        x = F.relu(self.bn1(self.conv1(s)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.fc1(x.view(x.size(0), -1))))
        x1 = F.relu(self.bn5(self.fc2(x)))
        x2 = F.relu(self.bn6(self.fc3(a)))
        x = torch.cat([x1, x2], dim=1)
        x = self.fc4(x)
        return x

class Agent(object):
    def __init__(self, state_dim, action_dim,
                 tau=0.001, discount=0.99,
                 lr_a=1e-4, lr_c=1e-3):
        self.discount = discount
        self.tau = tau

        self.actor = Actor(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_a)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_c, weight_decay=0.01)

        # Sync the initial weights
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        if USE_CUDA:
            self.actor.cuda()
            self.target_actor.cuda()
            self.critic.cuda()
            self.target_critic.cuda()

    def predict(self, s, epsilon=0.0):
        a = to_numpy(self.actor.forward(Variable(s).type(FLOAT))).squeeze(1)
        a += OU_noise(a, 0.0, epsilon)
        a = np.clip(a, -1., 1.)
        return a

    def train(self, batch):
        s, a, r, s2, term = batch[0], batch[1], batch[2], batch[3], batch[4]
        q2 = self.target_critic.forward(Variable(s2, volatile=True).type(FLOAT),
                                       self.target_actor.forward(Variable(s2, volatile=True).type(FLOAT)))
        q2.volatile = False
        term = [not e for e in term]
        y = to_tensor(r) + self.discount * to_tensor(np.array(term).astype(np.float)) * q2
        self.critic.zero_grad()
        q = self.critic.forward(Variable(s).type(FLOAT), to_tensor(a))
        c_loss = MSELoss_fn(q, y)
        c_loss.backward()
        self.critic_optimizer.step()

        self.actor.zero_grad()
        a_loss = -self.critic.forward(Variable(s).type(FLOAT),
                                      self.actor.forward(Variable(s).type(FLOAT)))
        a_loss = a_loss.mean()
        a_loss.backward()
        self.actor_optimizer.step()

    def train_target(self):
        for t, s in zip(self.target_actor.parameters(), self.actor.parameters()):
            t.data.copy_((1 - self.tau) * t.data + self.tau * s.data)
        for t, s in zip(self.target_critic.parameters(), self.critic.parameters()):
            t.data.copy_((1 - self.tau) * t.data + self.tau * s.data)

    def save(self, path):
        torch.save(self.actor, path + 'actor.pkl')
        torch.save(self.target_actor, path + 'target_actor.pkl')
        torch.save(self.critic, path + 'critic.pkl')
        torch.save(self.target_critic, path + 'target_critic.pkl')

class ReplayMemory(object):
    """docstring for ReplayBuffer"""
    def __init__(self, size, random_seed=123):
        self.size = size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, s2, t):
        experience = (s,a,r,s2,t)
        if self.count < self.size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def get_minibatch(self, batch_sz):
        if self.count < batch_sz:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_sz)

        s = torch.cat([e[0] for e in batch], dim=0)
        a = np.array([e[1] for e in batch])
        r = np.array([e[2] for e in batch])
        s2 = torch.cat([e[3] for e in batch], dim=0)
        t = np.array([e[4] for e in batch])
        return (s, a, r, s2, t)

class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """
    def _action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def _reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)


# Experiment pipeline
env = NormalizedEnv(gym.make('Pendulum-v0'))
state_dim = [9, 64, 64]
action_dim = env.action_space.shape[0]
agent = Agent(state_dim, action_dim)
memory = ReplayMemory(size=100000)

step = episode = episode_step = 0
scale = 10000
num_steps = 20*scale
epsilon = 1.0
epsilon_end = 0.1
epsilon_endt = 13*scale
warmup = 20
batch_sz = 16
image_stack = action_repeat = 3
image_holder = deque(maxlen=image_stack)
reward_holder = []
episode_reward = 0
s = None
rewards = []
path = 'outputs/'
plt.ion()

print 'Start of training...'
print '-'*20
while step < num_steps:
    eps = max(epsilon_end, max(0.0, epsilon - step / float(epsilon_endt)))

    if s is None:
        env.reset()
        image = get_screen(env)
        for _ in range(image_stack):
            image_holder.append(image)
        s = torch.cat(list(image_holder), dim=1)

    a = agent.predict(s, eps)
    for _ in range(action_repeat):
        obs, r_, term, info = env.step(a)
        reward_holder.append(r_)
        image_holder.append(get_screen(env))
        if term:
            break

    r = np.mean(reward_holder)
    s2 = torch.cat(list(image_holder), dim=1) # s2 actually not used if term == True

    memory.add(s, a, r, s2, term)
    if step > warmup:
        batch = memory.get_minibatch(batch_sz)
        agent.train(batch)
        agent.train_target()

    step += 1
    episode_step += 1
    episode_reward += r
    reward_holder = []
    s = s2
    if term:
        print '# %3d: ep_r: %.3f, epsilon: %.4f, steps: %5d' % (episode + 1, episode_reward, eps, step)
        rewards.append(episode_reward)
        s = None
        episode_reward = 0
        episode_step = 0
        episode += 1
        image_holder.clear()

        if episode % 10 == 9:
            if not os.path.exists(path):
                os.makedirs(path)
            agent.save(path)

        plt.cla()
        plt.plot(np.arange(len(rewards)), np.array(rewards), lw=1.5)
        plt.pause(0.1)

plt.ioff()
plt.xlabel('Number of episodes')
plt.ylabel('Episode score')
plt.savefig(path + 'ep_rewards.png')


