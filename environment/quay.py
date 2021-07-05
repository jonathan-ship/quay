import simpy
import numpy as np

from simulation import *


class QuayScheduling:
    def __init__(self, df_quay, df_work, df_score, df_ship, df_work_fix, log_path):
        self.df_quay = df_quay
        self.df_work = df_work
        self.df_score = df_score
        self.df_ship = df_ship
        self.df_work_fix = df_work_fix
        self.log_path = log_path
        self.done = False

        self.sim_env, self.ships, self.quays, self.monitor = self._modeling()

    def step(self, action):
        # simulation 진행
        reward = self._calculate_reward()
        next_state = self._get_State()
        return next_state, reward, self.done

    def reset(self):
        self.sim_env, self.ships, self.quays, self.monitor = self._modeling()
        self.done = False

        while True:
            if self.ships[0].current_work.start == self.sim_env.now:
                break
            self.sim_env.step()

        return self._get_State()

    def _get_State(self):
        # 배정된 선박의 안벽 선택 정보
        f_1 = np.zeros(len(self.df_score.columns)+2)
        # 해당 선박을 그 안벽에 집어 넣을 수 있는지 (자르기 여부까지 고려)
        f_2 = np.zeros(len())
        #
        f_3 = np.zeros()

        state = np.concatenate()
        return state

    def _calculate_reward(self):
        reward = 0
        pass
        return reward

    def _modeling(self):
        sim_env = simpy.Environment()  # 시뮬레이션 환경 생성
        monitor = Monitor(self.log_path)  # 시뮬레이션 이벤트 로그 기록을 위한 Monitor 객체 생성

        # 안벽 배치 대상 선박에 대한 리스트 생성
        ships = []
        for i, row in self.df_ship.iterrows():
            works = self.df_work[self.df_work["선종"] == row["선종"]]  # 선박의 선종에 해당하는 안벽 작업 데이터프레임 선택
            # 선박의 작업 중 안벽 고정 작업이 있는 지 확인
            if row["호선번호"] in self.df_work_fix["호선번호"]:
                # 고정 작업이 있을 경우
                fix_name = self.df_work_fix[self.df_work_fix["호선번호"] == row["호선번호"]]["작업명"]  # 고정 작업명
                fix_idx = int(works[works["작업"] == fix_name]["순번"]) - 1  # 선박의 작업 리스트 중 고정 작업의 순서 index
            else:
                # 고정 작업이 없을 경우
                fix_idx = 0

            # Ship 클래스에 대한 객체 생성 후 선박 리스트에 추가
            ship = Ship(row["호선번호"], row["선종"], row["길이"], row["진수일"], row["인도일"], works, fix_idx=fix_idx)
            ships.append(ship)

        # 시뮬레이션 프로세스 모델링
        quays = {}
        quays["Source"] = Source(sim_env, ships, quays, monitor)  # Source
        for i, row in self.df_quay.iterrows():
            scores = df_score[row["안벽"]]
            quay = Quay(sim_env, row["안벽"], quays, row["길이"], scores, monitor)
            quays[row["안벽"]] = quay  # Quay
        quays["S"] = Sea(sim_env, quays, monitor)  # Sea
        quays["Sink"] = Sink(sim_env, quays, monitor)  # Sink

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