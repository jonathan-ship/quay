import os

from environment.data import import_train_data
from environment.quay import QuayScheduling

import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
learning_rate = 0.0005
gamma = 0.8
buffer_limit = 100000
batch_size = 32


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self, state_size, action_size):
        super(Qnet, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def sample_action(self, obs, possible_action, epsilon):
        out = self.forward(obs)
        out = out.detach().numpy()
        mask = np.array(possible_action)
        #print(mask)
        out[mask==False] = float('-inf')
        #print(out)


        coin = np.random.uniform(0, 1)
        if coin < epsilon:
            #print(np.where(out != float('-inf'))[0])
            return np.random.choice(np.where(out != float('-inf'))[0])
        else:
            return np.argmax(out)


def train(q, q_target, memory, optimizer):
    for i in range(1):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



# if __name__ == "__main__":
#
#     info_path = "./data/기준정보+위탁과제용.xlsx"
#     scenario_path = "./data/[수정] 호선일정정보+위탁과제.xlsx"
#
#     log_path = '../result/log.csv'
#     if not os.path.exists(log_path):
#         os.makedirs(log_path)
#
#     df_quay, df_work, df_score, df_ship, df_work_fix = import_data(info_path, scenario_path)
#
#     env = QuayScheduling(df_quay, df_work, df_score, df_ship, df_work_fix, log_path)
#     done = False
#     state = env.reset()
#     r = []
#     while not done:
#         action =



if __name__ == "__main__":
    import pandas as pd
    from environment.data import *

    info_path = "./data/기준정보.xlsx"
    scenario_path = "./data/호선정보_학습.xlsx"
    #agent = Qnet(sa)

    log_path = '../result/log.csv'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    df_quay, df_score, df_weight, df_ship, df_work, df_work_fix = import_train_data(info_path, scenario_path)
    env = QuayScheduling(df_quay, df_score, df_weight, df_ship, df_work, df_work_fix, log_path)
    total_quay_list = env.df_quay['안벽'].tolist()
    state_size = 85
    action_size = 29
    q = Qnet(state_size, action_size)
    q_target = Qnet(state_size, action_size)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    optimizer = optim.RMSprop(q.parameters(), lr=0.00008, alpha=0.99, eps=1e-06)








    update_interval = 20
    score = 0

    step = 0
    moving_average = []
    for n_epi in range(10000):
        done = False
        state = env.reset()
        r = []
        while done == False:

            epsilon = max(0.01, 0.1-0.01*(n_epi/200))
            possible_action = env.model["Routing"].possible_quay
            possible_action = [env.inverse_mapping[key] for key in possible_action.keys()]
            avail_action = [True if i in possible_action else False for i in range(29)]
            #print(avail_action)
            step+=1
            action = q.sample_action(torch.from_numpy(state).float(), avail_action, epsilon)
            # print(action)
            #np.random.choice(possible_action)
            #action = np.random.randint(29)
            done_mask = 0.0 if done else 1.0
            next_state, reward, done = env.step(action)
            #print(reward)
            r.append(reward)
            memory.put((state, action, reward, next_state, done_mask))

            if memory.size()>2000:
                train(q, q_target, memory, optimizer)

            state = next_state

            if n_epi % update_interval == 0 and n_epi != 0:
                q_target.load_state_dict(q.state_dict())

            if done == True:
                print(sum(r), epsilon)
                moving_average.append(sum(r))
                df = pd.DataFrame(moving_average)
                df.to_csv("moving_average.csv")
                break
        #
        # print(env.model["Sink"].ships_rec)
        # print(env.model["Routing"].ship.name)
        # print(env.model["Routing"].possible_quay)