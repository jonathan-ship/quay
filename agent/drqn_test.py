import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
from torch.autograd import Variable

from analyze import analyze
from drqn import *


if __name__ == '__main__':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    from environment.data import import_train_data
    from environment.quay import QuayScheduling
    import pandas as pd
    from environment.data import *
    info_path = "../environment/data/기준정보.xlsx"

    scenario_path_test_high_load = "../environment/data/호선정보_학습.xlsx"
    log_path_test_high_load = '../result/log_test_high_load/'
    if not os.path.exists(log_path_test_high_load):
        os.makedirs(log_path_test_high_load)
    df_quay_test, df_score_test, df_weight_test, df_ship_test, df_work_test, df_work_fix_test, start_test = import_train_data(info_path, scenario_path_test_high_load)
    env_test = QuayScheduling(df_quay_test, df_score_test, df_weight_test, df_ship_test, df_work_test, df_work_fix_test, log_path_test_high_load)
    print("액티비티 수(개) : %d" % len(df_work_test))
    # scenario_path_test = "../environment/data/호선정보_테스트_low_load.xlsx"
    # log_path_test_low_load = '../result/log_test_low_load/'

    model_path = '../result/model/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # if not os.path.exists(log_path_test_low_load):
    #     os.makedirs(log_path_test_low_load)

    # df_quay_test_2, df_score_test_2, df_weight_test_2, df_ship_test_2, df_work_test_2, df_work_fix_test_2, start_test_2 = import_train_data(info_path, scenario_path_test)
    # env_test_2 = QuayScheduling(df_quay_test_2, df_score_test_2, df_weight_test_2, df_ship_test_2, df_work_test_2, df_work_fix_test_2, log_path_test_low_load)

    random.seed()

    #env = QuayScheduling(df_quay, df_score, df_weight, df_ship, df_work, df_work_fix, log_path)
    state_size = 265
    action_size = 29
    hidden_size = 32
    max_epi_num = 1000
    max_epi_len = 5000
    batch_size = 3
    eval_freq = 10
    train_start_epi = 100
    agent = Agent(state_size = state_size, action_size = action_size, hidden_size = hidden_size, batch_size = batch_size, max_epi_num = max_epi_num, max_epi_len = max_epi_len, env = None)
    #agent.drqn_tar.load_state_dict(agent.drqn.state_dict())

    train_curve = []
    epi = 8200

    ### SCENARIO_1 TEST
    agent.drqn.load_state_dict(torch.load(model_path+"weight_{}.pt".format(epi)))

    start = time.time()

    state_test = env_test.reset()
    done_test = False
    hidden_state_test = torch.zeros((1, 1, hidden_size))
    rewards_test = []
    while done_test == False:
        possible_action_test = env_test.model["Routing"].possible_quay
        possible_action_test = [env_test.inverse_mapping[key] for key in possible_action_test.keys()]
        avail_action_test = [True if i in possible_action_test else False for i in range(29)]
        action_test, hidden_state_test = agent.get_action(torch.from_numpy(state_test).float().unsqueeze(0).unsqueeze(0), hidden_state_test, avail_action_test, 0)
        next_state_test, reward_test, done_test = env_test.step(action_test)
        rewards_test.append(reward_test)
        state_test = next_state_test

        if done_test:
            finish = time.time()
            print("시뮬레이션 실행시간(s) : %f" % (finish - start))
            print("분당 처리 수 : %f" % (len(df_work_test) * 60 / (finish - start)))
            print("시나리오1 ", 'reward', sum(rewards_test), '전문안벽 배치율', np.sum([ship.quay_cnt for ship in env_test.ships])/np.sum([ship.total_time for ship in env_test.ships]))

            quay_arrangement = dict()
            for ship in env_test.ships:
                quay_arrangement[ship.name] = ship.quay_cnt_list
            df_quay_arrangement = pd.DataFrame(quay_arrangement)
            df_T = df_quay_arrangement.transpose()
            df_T.columns=["A", "B", "C", "D", "E", "N"]
            df_T.to_csv("../result/log_test_high_load/quay_arrangement_durations_by_level_high_load.csv")
            # input_path = '../result/log_test_high_load/log_{}.csv'.format(env_test.e)
            # analyze(input_path, load = 'high_load')
            break

    # ### SCENARIO_2 TEST
    # agent.drqn.load_state_dict(torch.load(model_path+"weight_{}.pt".format(epi)))
    # state_test = env_test_2.reset()
    # done_test = False
    # hidden_state_test = torch.zeros((1, 1, hidden_size))
    # rewards_test = []
    # while done_test == False:
    #     possible_action_test = env_test_2.model["Routing"].possible_quay
    #     possible_action_test = [env_test_2.inverse_mapping[key] for key in possible_action_test.keys()]
    #     avail_action_test = [True if i in possible_action_test else False for i in range(29)]
    #     action_test, hidden_state_test = agent.get_action(torch.from_numpy(state_test).float().unsqueeze(0).unsqueeze(0), hidden_state_test, avail_action_test, 0)
    #     next_state_test, reward_test, done_test = env_test_2.step(action_test)
    #     rewards_test.append(reward_test)
    #     state_test = next_state_test
    #     if done_test:
    #         print("시나리오2 ",'reward', sum(rewards_test), '전문안벽 배치율', np.sum([ship.quay_cnt for ship in env_test_2.ships])/np.sum([ship.total_time for ship in env_test_2.ships]))
    #
    #         quay_arrangement = dict()
    #         for ship in env_test_2.ships:
    #             quay_arrangement[ship.name] = ship.quay_cnt_list
    #         df_quay_arrangement = pd.DataFrame(quay_arrangement)
    #         df_T = df_quay_arrangement.transpose()
    #         df_T.columns=["A", "B", "C", "D", "E", "N"]
    #
    #         df_T.to_csv("../result/log_test_low_load/quay_arrangement_durations_by_level_low_load.csv")
    #
    #         input_path = '../result/log_test_low_load/log_{}.csv'.format(env_test_2.e)
    #         analyze(input_path, load='low_load')
    #         break

