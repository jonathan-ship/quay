import simpy

from simulation import *


class QuayScheduling:
    def __init__(self, df_quay, df_work, df_score, df_ship, df_work_fix, log_path):
        self.df_quay = df_quay
        self.df_work = df_work
        self.df_score = df_score
        self.df_ship = df_ship
        self.df_work_fix = df_work_fix
        self.log_path = log_path

        self.sim_env, self.ships, self.quays, self.monitor = self._modeling()

    def step(self):
        pass

    def reset(self):
        pass

    def _get_State(self):
        pass

    def _calculate_reward(self):
        pass

    def _modeling(self):
        sim_env = simpy.Environment()
        monitor = Monitor(self.log_path)

        ships = []
        for i, row in self.df_ship.iterrows():
            works = self.df_work[self.df_work["선종"] == row["선종"]]
            if row["호선번호"] in self.df_work_fix["호선번호"]:
                fix_name = self.df_work_fix[self.df_work_fix["호선번호"] == row["호선번호"]]["작업명"]
                fix_idx = int(works[works["작업"] == fix_name]["순번"]) - 1
            else:
                fix_idx = 0

            ship = Ship(row["호선번호"], row["선종"], row["길이"], row["진수일"], row["인도일"], works, fix_idx=fix_idx)
            ships.append(ship)

        quays = {}
        for i, row in self.df_quay.iterrows():
            scores = df_score[row["안벽"]]
            quay = Quay(row["안벽"], row["길이"], scores)
            quays[row["안벽"]] = quay
        quays["Sink"] = Sink(env, quays, monitor)

        return sim_env, ships, quays, monitor


if __name__ == "__main__":

    from data import import_data

    info_path = "./data/기준정보+위탁과제용.xlsx"
    scenario_path = "./data/[수정] 호선일정정보+위탁과제.xlsx"

    log_path = '../result/log.csv'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    df_quay, df_work, df_score, df_ship, df_work_fix = import_data(info_path, scenario_path)

    env = QuayScheduling(df_quay, df_work, df_score, df_ship, df_work_fix, log_path)
    print("d")