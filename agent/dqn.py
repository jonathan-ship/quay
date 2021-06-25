import os

from environment.data import import_data
from environment.quay import QuayScheduling

if __name__ == "__main__":

    info_path = "./data/기준정보+위탁과제용.xlsx"
    scenario_path = "./data/[수정] 호선일정정보+위탁과제.xlsx"

    log_path = '../result/log.csv'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    df_quay, df_work, df_score, df_ship, df_work_fix = import_data(info_path, scenario_path)

    env = QuayScheduling(df_quay, df_work, df_score, df_ship, df_work_fix, log_path)