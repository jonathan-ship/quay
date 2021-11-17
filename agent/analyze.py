import pandas as pd


def analyze(input_path, load):
    df = pd.read_csv(input_path)
    df = df.set_index("Ship")
    index = set(df.index)
    if load == 'high_load':
        criteria = pd.read_excel("../environment/data/호선정보_테스트_high_load.xlsx", engine='openpyxl')
        criteria = criteria.set_index("호선번호")
        result = []
        for id in index:
            x = df.loc[id]
            temp = 0
            for t in range(len(x)):
                try:
                    if x.iloc[t, 3] != x.iloc[t + 1, 3]:
                        temp += 1
                except IndexError:
                    pass
            # crit = criteria.loc[id, "최대이동횟수"]

            type = criteria.loc[id, "선종"]
            if type == 'LNG':
                limit = 9
            elif type == 'PC':
                limit = 5
            else:
                limit = 4
            result.append([id, type, temp, limit, temp - limit])

        final = pd.DataFrame(result, columns=["호선", "선종", "학습이동횟수", "이동한계", "초과량"])
        final.to_csv("../result/log_test_high_load/movement_{}_.csv".format(load), encoding='utf-8-sig')
    else:
        criteria = pd.read_excel("../environment/data/호선정보_테스트_low_load.xlsx", engine='openpyxl')
        criteria = criteria.set_index("호선번호")
        result = []
        for id in index:
            x = df.loc[id]
            temp = 0
            for t in range(len(x)):
                try:
                    if x.iloc[t, 3] != x.iloc[t + 1, 3]:
                        temp += 1
                except IndexError:
                    pass
            # crit = criteria.loc[id, "최대이동횟수"]

            type = criteria.loc[id, "선종"]
            if type == 'LNG':
                limit = 9
            elif type == 'PC':
                limit = 5
            else:
                limit = 4
            result.append([id, type, temp, limit, temp - limit])

        final = pd.DataFrame(result, columns=["호선", "선종", "학습이동횟수", "이동한계", "초과량"])
        final.to_csv("../result/log_test_low_load/movement_{}_.csv".format(load), encoding='utf-8-sig')

