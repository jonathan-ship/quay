from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
import numpy as np
import math
from analyze import *

class ReplayMemory(object):
    def __init__(self, max_epi_num, max_epi_len):
        # capacity is the maximum number of episodes
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.memory = deque(maxlen=self.max_epi_num)
        self.is_av = False
        self.current_epi = 0
        self.memory.append([])

    def reset(self):

        self.current_epi = 0
        self.memory.clear()
        self.memory.append([])

    def create_new_epi(self):
        self.memory.append([])
        self.current_epi = self.current_epi + 1
        if self.current_epi > self.max_epi_num - 1:
            self.current_epi = self.max_epi_num - 1

    def remember(self, state, action, reward, avail_action):
        self.memory[self.current_epi].append([state, action, reward, avail_action])

    def sample(self):
        #print(self.current_epi, len(self.memory))
        epi_index = random.randint(0, len(self.memory)-2)
        #print(self.memory[epi_index])
        #print(len(self.memory)-2)
        return self.memory[epi_index]


    def size(self):
        return len(self.memory)

    def is_available(self):
        self.is_av = True
        if len(self.memory) <= 1:
            self.is_av = False
        return self.is_av

    def print_info(self):
        for i in range(len(self.memory)):
            print('epi', i, 'length', len(self.memory[i])) #

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(1, -1)

class DRQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(DRQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.flat1 = Flatten()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.GRU_layer = nn.GRU(64, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 32)
        self.fc4 = nn.Linear(32, self.action_size)

    def forward(self, x, hidden):
        x.float()
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        gru_out, hidden = self.GRU_layer(h2, hidden)
        h4 = F.relu(self.fc3(gru_out))
        h5 = self.fc4(h4)
        return h5, hidden

class Agent(object):
    def __init__(self, state_size, action_size, hidden_size, batch_size, max_epi_num, max_epi_len, env):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.env = env

        self.drqn = DRQN(state_size, action_size, hidden_size)
        self.drqn_tar = DRQN(state_size, action_size, hidden_size)
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.buffer = ReplayMemory(max_epi_num=self.max_epi_num, max_epi_len=self.max_epi_len)

        "훈련과 관련된 hyperparameter"
        self.gamma = 0.9
        self.TAU = 1e-2
        self.LR = 1e-4


        self.optimizer = torch.optim.Adam(self.drqn.parameters(), lr=self.LR)

    def remember(self, state, action, reward, avail_action):
        self.buffer.remember(state, action, reward, avail_action)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)


    def train(self):
        losses = []
        for _ in range(self.batch_size):
            memo = self.buffer.sample()
            obs_list = []
            action_list = []
            reward_list = []
            avail_action_list = []
            for i in range(len(memo)):
                obs_list.append(memo[i][0].tolist())
                action_list.append(memo[i][1])
                reward_list.append(memo[i][2])
                avail_action_list.append(memo[i][3])

            states = torch.tensor(obs_list).view(-1, 1, self.state_size)
            hidden = torch.zeros((1, 1, self.hidden_size))
            hidden_cur = torch.zeros((1, 1, self.hidden_size))
            Q_cur, hidden_cur = self.drqn.forward(states, hidden_cur)
            Q, hidden = self.drqn_tar.forward(states, hidden)
            Q_tar = Q.clone()
            mask = np.array(avail_action_list).reshape(-1, 1, self.action_size)
            Q[mask==False] = float('-inf')

            for t in range(len(memo) - 1):
                max_next_q = torch.max(Q_tar[t+1, 0, :]).clone().detach()
                Q_tar[t, 0, action_list[t]] = reward_list[t] + self.gamma * max_next_q
            T = len(memo) - 1
            Q_tar[T, 0, action_list[T]] = reward_list[T]
            i = torch.tensor(action_list).view(-1, 1, 1)
            q_cur = Q_cur.gather(index = i, dim = 2)
            q_tar = Q_tar.gather(index = i, dim = 2)

            loss = F.smooth_l1_loss(q_cur, q_tar)
            losses.append(loss.detach().item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.soft_update(self.drqn, self.drqn_tar)
        return np.mean(losses)

    def get_action(self, obs, hidden, possible_action, epsilon):

        out, hidden = self.drqn.forward(obs, hidden)
        out = out.detach().numpy().reshape(-1)
        mask = np.array(possible_action)
        out[mask==False] = float('-inf')
        coin = np.random.uniform(0, 1)
        if coin < epsilon:
            return np.random.choice(np.where(out != float('-inf'))[0]), hidden
        else:
            return np.argmax(out), hidden

def get_decay(epi_iter):
    decay = math.pow(0.999, epi_iter)
    if decay < 0.005:
        decay = 0.01
    return decay


if __name__ == '__main__':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    from environment.data import import_train_data
    from environment.quay import QuayScheduling
    import pandas as pd
    from environment.data import *

    info_path = "../environment/data/기준정보.xlsx" # 기준정보 경로
    scenario_path = "../environment/data/호선정보_학습.xlsx" # 학습을 위한 호선정보 경로

    log_path = '../result/log/' # event_log가 저장되는 경로
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    model_path = '../result/model/'  # event_log가 저장되는 경로
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    df_quay, df_score, df_weight, df_ship, df_work, df_work_fix, start = import_train_data(info_path, scenario_path) # 전처리
    env = QuayScheduling(df_quay, df_score, df_weight, df_ship, df_work, df_work_fix, log_path) # 환경(시뮬레이션) 객체 생성

    random.seed()

    ### Agent 객체 생성을 위해 필요한 정보
    state_size = 265
    action_size = 29
    hidden_size = 32 # hidden_state의 사이즈
    max_epi_num = 1000 # replay_memory에 저장되는 에피소드는 1000에피소드임(rnn을 사용하여, 에피소드 단위로 저장이됨)
    max_epi_len = 5000 # 불필요한 파라미터
    batch_size = 3 # train을 시작할 당시의 mini_batch 크기는 3
    train_start_epi = 100 # train_start_epi 전까지는 replay-memory에 s,a,r,s' 단위로 저장하기만 한다.

    ### Agent 객체 생성
    agent = Agent(state_size = state_size, action_size = action_size, hidden_size = hidden_size, batch_size = batch_size, max_epi_num = max_epi_num, max_epi_len = max_epi_len, env = None)
    agent.drqn_tar.load_state_dict(agent.drqn.state_dict()) # target_network가 current network를 복사한다.

    ### 훈련을 수행
    for epi_iter in range(200000):
        state = env.reset()
        done = False
        hidden_state = torch.zeros((1, 1, hidden_size))
        rewards = []
        while done == False:
            possible_action = env.model["Routing"].possible_quay
            possible_action = [env.inverse_mapping[key] for key in possible_action.keys()]
            avail_action = [True if i in possible_action else False for i in range(29)]
            action, hidden_state = agent.get_action(torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0), hidden_state, avail_action, get_decay(epi_iter))
            next_state, reward, done = env.step(action)
            rewards.append(reward)
            agent.remember(state, action, reward, avail_action)
            state = next_state
            if done:
                agent.buffer.create_new_epi()
                if epi_iter % 100 == 0: # 100 에피소드 단위로
                    torch.save(agent.drqn.state_dict(), model_path + "/weight_{}.pt".format(epi_iter))
                break
        if epi_iter > train_start_epi:
            if epi_iter > train_start_epi*2:
                agent.batch_size = 5
            if epi_iter > train_start_epi*5:
                agent.batch_size = 12
            loss = agent.train()
            print('Episode', epi_iter, 'reward', sum(rewards), 'loss', loss)
        else:
            print('Episode', epi_iter, 'reward', sum(rewards))