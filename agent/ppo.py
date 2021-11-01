import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datetime import datetime
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()

        self.fc1 = nn.Linear(state_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.policy = nn.Linear(256, action_size)
        self.value = nn.Linear(256, 1)

        self._initialize_weights()

    def _initialize_weights(self, init_min=-0.08, init_max=0.08):
        for param in self.parameters():
            nn.init.uniform_(param.data, init_min, init_max)

    def pi(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.policy(x)
        return x

    def v(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.value(x)
        return x


class PPOAgent():
    def __init__(self, env, log_dir, model_dir, load_model=False):
        self.env = env
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.log_dir = log_dir
        self.model_dir = model_dir

        self.date = datetime.now().strftime('%m%d_%H_%M')
        self.log_path = self.log_dir + 'ppo' + '/%s_train.csv' % self.date
        with open(self.log_path, 'w') as f:
            f.write('episode, reward, average loss\n')

        self.avg_loss = 0.0

        self.learning_rate = 1e-5
        self.epsilon = 0.2
        self.gamma = 0.75
        self.lmbda = 0.95
        self.K_epoch = 2
        self.T = 20

        self.start_episode = 0
        self.model = ActorCritic(env.state_size, env.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if load_model:
            checkpoint = torch.load(self.model_dir + "/ppo/" + max(os.listdir(self.model_dir + "/ppo")))
            self.start_episode = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.data = []

    def record(self, episode, reward_sum, avg_loss):
        with open(self.log_path, 'a') as f:
            f.write('%d,%1.4f,%1.4f\n' % (episode, reward_sum, avg_loss))

    def append_sample(self, state, action, reward, next_state, prob, mask, done):
        self.data.append((state, action, reward, next_state, prob, mask, done))

    def get_action(self, state):
        mask = torch.ones(self.env.action_size)
        possible_action = self.env.model["Routing"].possible_quay
        possible_action = [self.env.inverse_mapping[key] for key in possible_action.keys()]
        mask[possible_action] = 0.0

        logits = self.model.pi(torch.from_numpy(state).float())
        logits_maksed = logits - 1e8 * mask
        prob = torch.softmax(logits_maksed, dim=-1)
        m = Categorical(prob)
        action = m.sample().item()

        return action, mask, prob

    def copmute_loss(self):
        states = torch.tensor([sample[0] for sample in self.data], dtype=torch.float32)
        actions = torch.tensor([[sample[1]] for sample in self.data], dtype=torch.int64)
        rewards = torch.tensor([[sample[2]] for sample in self.data], dtype=torch.float32)
        next_states = torch.tensor([sample[3] for sample in self.data], dtype=torch.float32)
        probs = torch.tensor([[sample[4]] for sample in self.data], dtype=torch.float32)
        masks = torch.tensor([sample[5] for sample in self.data], dtype=torch.float32)
        dones = torch.tensor([[1 - sample[6]] for sample in self.data], dtype=torch.float32)

        td_target = rewards + self.gamma * self.model.v(next_states) * dones
        delta = td_target - self.model.v(states)

        advantage_lst = np.zeros(self.T)
        advantage = 0.0
        for t in reversed(range(0, len(delta))):
            advantage = self.gamma * self.lmbda * advantage + delta[t][0]
            advantage_lst[t] = advantage

        logits = self.model.pi(states)
        logits_maksed = logits - 1e8 * masks
        pi = torch.softmax(logits_maksed, dim=1)
        pi_a = pi.gather(1, actions)
        ratio = torch.exp(torch.log(pi_a) - torch.log(probs))  # a/b == exp(log(a)-log(b))

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.model.v(states), td_target.detach())

        self.avg_loss += loss.mean().detach()

        return loss

    def train_model(self):
        for i in range(self.K_epoch):
            loss = self.copmute_loss()
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        self.data = []

    def run(self, num_of_episodes):
        for e in range(self.start_episode + 1, num_of_episodes + 1):
            state = self.env.reset()
            done = False
            total_reward = 0.0
            total_move = 0.0
            total_score = 0.0

            step = 0
            while not done:
                for t in range(self.T):
                    step += 1

                    action, mask, prob = self.get_action(state)
                    next_state, reward, done = self.env.step(action)

                    self.append_sample(state, action, reward, next_state, prob[action], mask.numpy(), done)

                    state = next_state
                    total_reward += reward
                    total_move += self.env.reward_move
                    total_score += self.env.reward_efficiency

                    if done:
                        break

                self.train_model()

            print('epside:%d/%d, reward:%1.3f, average loss:%1.3f' %
                  (e, num_of_episodes, total_reward, self.avg_loss / (self.K_epoch * step)))
            self.record(e, total_reward, self.avg_loss / (self.K_epoch * step))

            if e % 100 == 0:
                torch.save({'episode': e,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                           self.model_dir + '/ppo' + '/%s_episode%d.pt' % (self.date, e))
                print('save model...')