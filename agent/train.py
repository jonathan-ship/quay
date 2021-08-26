import os

from agent.ppo import PPOAgent
from environment.data import import_train_data
from environment.quay import QuayScheduling


if __name__ == "__main__":

    model = "ppo"

    info_path = "../environment/data/기준정보.xlsx"
    scenario_path = "../environment/data/호선정보_학습.xlsx"

    event_dir = '../result/event/'
    if not os.path.exists(event_dir):
        os.makedirs(event_dir)

    log_dir = '../result/log/'
    if not os.path.exists(log_dir + model):
        os.makedirs(log_dir + model)

    model_dir = '../result/model/'
    if not os.path.exists(model_dir + model):
        os.makedirs(model_dir + model)

    df_quay, df_score, df_weight, df_ship, df_work, df_work_fix, start = import_train_data(info_path, scenario_path)
    env = QuayScheduling(df_quay, df_score, df_weight, df_ship, df_work, df_work_fix, event_dir)

    num_of_episodes = 10000

    agent = PPOAgent(env, log_dir, model_dir)
    agent.run(num_of_episodes)