import pandas as pd


def import_data(info_path, scenario_path):
    df_quay = pd.read_excel(info_path, sheet_name="안벽 기준정보", header=1, engine='openpyxl')
    df_work = pd.read_excel(info_path, sheet_name="작업 기준정보", header=1, engine='openpyxl')
    df_score = pd.read_excel(info_path, sheet_name="안벽작업 기준정보", header=1, engine='openpyxl')

    df_quay = df_quay.dropna(axis=1)
    df_quay = df_quay[df_quay["안벽"].map(lambda x: not "S" in x)]
    df_work = df_work.dropna(axis=1)
    df_score = df_score.dropna(axis=1)
    df_score = df_score.set_index(["선종", "작업"])

    df_ship = pd.read_excel(scenario_path, sheet_name="호선정보", engine='openpyxl')
    df_work_fix = pd.read_excel(scenario_path, sheet_name="고정작업정보", engine='openpyxl')
    df_ship = df_ship[df_ship["선종"] != "OTHERS"]
    df_work_fix = df_work_fix[df_work_fix["선종"] != "OTHERS"]

    return df_quay, df_work, df_score, df_ship, df_work_fix
