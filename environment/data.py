import numpy as np
import pandas as pd
import plotly.express as px


def import_train_data(info_path, scenario_path):
    df_quay = pd.read_excel(info_path, sheet_name="안벽 기준정보", engine='openpyxl')
    df_work_temp = pd.read_excel(info_path, sheet_name="작업 기준정보", engine='openpyxl')
    df_score = pd.read_excel(info_path, sheet_name="안벽작업 기준정보", engine='openpyxl')
    df_weight = pd.read_excel(info_path, sheet_name="가중치 정보", engine='openpyxl')

    df_ship = pd.read_excel(scenario_path, sheet_name="호선리스트", engine='openpyxl')
    df_work = pd.read_excel(scenario_path, sheet_name="작업리스트", engine='openpyxl')
    df_work_fixed = pd.read_excel(scenario_path, sheet_name="고정작업리스트", engine='openpyxl')

    df_sharing = pd.DataFrame([["B2", "B3"], ["C1", "C3"], ["C2", "C4"], ["D3", "D5"], ["D1", "D2", "D4"]],
                              columns=["공유1", "공유2", "공유3"])

    df_quay["공유1"] = None
    df_quay["공유2"] = None
    df_quay["공유3"] = None
    for i, row in df_quay.iterrows():
        shared_quay_set = []

        temp = df_sharing[(df_sharing["공유1"] == row["안벽"]) |
                          (df_sharing["공유2"] == row["안벽"]) |
                          (df_sharing["공유3"] == row["안벽"])].dropna(axis=1)
        if len(temp) == 0:
            shared_quay_set.append(row["안벽"])
        else:
            temp_trs = temp.values.tolist()[0]
            if len(temp_trs) == 2:
                shared_quay_set.extend(temp_trs)
            elif len(temp_trs) == 3:
                idx = temp_trs.index(row["안벽"])
                if idx == 0:
                    shared_quay_set.extend(temp_trs[:2])
                elif idx == 1:
                    shared_quay_set.extend(temp_trs)
                else:
                    shared_quay_set.extend(temp_trs[1:])

        for j, quay in enumerate(shared_quay_set):
            df_quay.loc[i, "공유{0}".format(j + 1)] = quay

    df_score = df_score.set_index(["선종", "작업"])
    df_weight = df_weight.set_index(["선종"])

    df_ship['진수일'] = pd.to_datetime(df_ship['진수일'], format='%Y-%m-%d')
    df_ship['인도일'] = pd.to_datetime(df_ship['인도일'], format='%Y-%m-%d')
    df_work['시작일'] = pd.to_datetime(df_work['시작일'], format='%Y-%m-%d')
    df_work['종료일'] = pd.to_datetime(df_work['종료일'], format='%Y-%m-%d')
    df_work_fixed['시작일'] = pd.to_datetime(df_work_fixed['시작일'], format='%Y-%m-%d')
    df_work_fixed['종료일'] = pd.to_datetime(df_work_fixed['종료일'], format='%Y-%m-%d')
    start = df_ship["진수일"].min()
    df_ship['진수일'] = (df_ship['진수일'] - start).dt.days
    df_ship['인도일'] = (df_ship['인도일'] - start).dt.days
    df_work['시작일'] = (df_work['시작일'] - start).dt.days
    df_work['종료일'] = (df_work['종료일'] - start).dt.days
    df_work_fixed['시작일'] = (df_work_fixed['시작일'] - start).dt.days
    df_work_fixed['종료일'] = (df_work_fixed['종료일'] - start).dt.days

    df_ship = df_ship.sort_values(by=["진수일"])
    df_ship = df_ship.reset_index(drop=True)

    df_work["자르기"] = df_work.apply(
        lambda x: df_work_temp[(df_work_temp["선종"] == x["선종"])
                               & (df_work_temp["작업"] == x["작업명"])]["자르기"].tolist()[0], axis=1)
    df_work["필수기간"] = df_work.apply(
        lambda x: df_work_temp[(df_work_temp["선종"] == x["선종"])
                               & (df_work_temp["작업"] == x["작업명"])]["필수기간"].tolist()[0], axis=1)

    return df_quay, df_score, df_weight, df_ship, df_work, df_work_fixed, start


def import_test_data(info_path, scenario_path):
    df_quay = pd.read_excel(info_path, sheet_name="안벽 기준정보", header=1, engine='openpyxl')
    df_work = pd.read_excel(info_path, sheet_name="작업 기준정보", header=1, engine='openpyxl')
    df_score = pd.read_excel(info_path, sheet_name="안벽작업 기준정보", header=1, engine='openpyxl')
    df_score = df_score.set_index(["선종", "작업"])

    df_ship = pd.read_excel(scenario_path, sheet_name="호선정보", engine='openpyxl')
    df_work_fix = pd.read_excel(scenario_path, sheet_name="고정작업정보", engine='openpyxl')
    df_ship = df_ship[df_ship["선종"] != "OTHERS"]
    df_work_fix = df_work_fix[df_work_fix["선종"] != "OTHERS"]

    return df_quay, df_work, df_score, df_ship, df_work_fix


def export_result(file_path, start):
    df_log = pd.read_csv(file_path)
    df_log["Time"] = df_log["Time"].map(lambda x: start + pd.Timedelta(days=x))

    df_gantt = pd.DataFrame(columns=["Task", "Start", "Finish", "Resource"])
    df_log_group = df_log.groupby(["Ship"])

    temp = 0
    for ship_name, group in df_log_group:
        group = group.sort_values(by=["Time"]).reset_index()
        for i in range(len(group)):
            if group.loc[i, "Quay"] == "Source":
                continue
            elif group.loc[i, "Quay"] == "Sink" or i == len(group) - 1:
                break
            else:
                if group.loc[i, "Time"] != group.loc[i + 1, "Time"]:
                    df_gantt.loc[temp] = [group.loc[i, "Quay"], group.loc[i, "Time"], group.loc[i + 1, "Time"], ship_name]
                    temp += 1

    fig = px.timeline(df_gantt, x_start="Start", x_end="Finish", y="Task", color="Resource")
    fig.update_yaxes(autorange="reversed")
    fig.show()
