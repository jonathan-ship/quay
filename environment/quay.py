import random

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
        self.move = 0

        self.sim_env, self.ships, self.quays, self.monitor = self._modeling()

    def step(self, action):
        # Take action at current decision time step
        quay_with_decision_time_step = []
        if self.quays['Source'].waiting:
            quay_with_decision_time_step.append('Source')
        for quay_name in list(self.df_quay['안벽']):
            if self.quays[quay_name].waiting:
                quay_with_decision_time_step.append(quay_name)
        ship_with_decision_time_step = list(self.quays['S'].waiting.keys())

        if quay_with_decision_time_step and not ship_with_decision_time_step:
            self.quays[quay_with_decision_time_step[0]].decision.succeed((self.df_quay['안벽'][action], True))
            self.quays[quay_with_decision_time_step[0]].waiting = False
        elif not quay_with_decision_time_step and ship_with_decision_time_step:
            self.quays["S"].decision[ship_with_decision_time_step[0]].succeed((self.df_quay['안벽'][action], True))
            del self.quays["S"].decision[ship_with_decision_time_step[0]]

        # Run until next decision time step
        while True:
            # decision_check_list = []
            # for quay in self.quays:
            #     if quay.name == "S":
            #         temp = quay.decision.keys()
            #     elif quay.name == "Sink":
            #         pass
            #     else :
            #         decision_check_list.append(quay.decision)
            # decision_check_list = decision_check_list + temp
            # unique = decision_check_list.unique()

            # Check whether there is any decision time step
            quay_with_decision_time_step = []
            if self.quays['Source'].waiting:
                quay_with_decision_time_step.append('Source')
            for quay_name in list(self.df_quay['안벽']):
                if self.quays[quay_name].waiting:
                    quay_with_decision_time_step.append(quay_name)
            ship_with_decision_time_step = list(self.quays['S'].waiting.keys())

            if quay_with_decision_time_step or ship_with_decision_time_step:
                break

            self.sim_env.step()

        reward = 4 #self._calculate_reward()
        next_state = 4 #self._get_state()

        if self.quays["Source"].sent == len(self.df_ship):
            self.done = True

        return next_state, reward, self.done

    def reset(self):
        self.sim_env, self.ships, self.quays, self.monitor = self._modeling()
        self.done = False
        self.move = 0

        # Go to first decision time step
        while True:
            # Check whether there is any decision time step
            quay_with_decision_time_step = []
            if self.quays['Source'].decision:
                quay_with_decision_time_step.append('Source')
            for quay_name in list(self.df_quay['안벽']):
                if self.quays[quay_name].decision:
                    quay_with_decision_time_step.append(quay_name)
            ship_with_decision_time_step = list(self.quays['S'].decision.keys())

            if quay_with_decision_time_step or ship_with_decision_time_step:
                break

            self.sim_env.step()

        return 3 #self._get_state()

    def _get_state(self):
        # 의사결정 시점에서 해당 작업의 각 안벽에 대한 선호도
        f_1 = np.zeros(len(self.df_score.columns)+2)
        # 해당 선박을 그 안벽에 집어 넣을 수 있는지 (자르기 여부까지 고려)
        f_2 = np.zeros(len(self.df_score.columns)+2)
        # 해당 안벽에 작업되고 있는 작업의 해당 안벽에서의 선호도
        f_3 = np.zeros(len(self.df_score.columns)+2)
        # 이동횟수 변수
        f_4 = np.zeros(1)


        for i, quay in enumerate(self.quays):
            ship_decision = None
            work_name = ship_decision.current_work.name  # 해당 안벽에서 작업중인 작업이름
            category = ship_decision.category
            f_1[i] = quay.score[category, work_name]
            if quay.queue.items != None:
                ship = quay.queue.items[0]  # 해당 안벽에 들어있는 ship 객체
                work_name = ship.current_work.name  # 해당 안벽에서 작업중인 작업이름
                category = ship.category    # 해당 안벽에서 작업중인 선박의 선종
                f_3[i] = quay.scores[category, work_name]    # 해당 안벽에서 작업중인 작업의 안벽에 대한 점수

                if ship.current_work.cut == 'S':
                    if ship.current_work.progress < ship.current_work.duration - ship.current_work.duration_fix:
                        f_2[i] = 1
                elif ship.current_work.cut == 'F':
                    if ship.current_work.progress > ship.current_work.duration - ship.current_work.duration_fix:
                        f_2[i] = 1
            else:
                f_2[i] = 2

        state = np.concatenate((f_1, f_2, f_3, f_4), axis = 0)
        return state

    def _calculate_reward(self):
        # 호선이동 횟수 역수할지 - 둘지 고민
        reward_move_no = self.move
        # 전문 안벽 배치율
        reward_prof_quay = 0
        for i, quay in enumerate(self.quays):
            if quay.score[quay.queue.items[0].category, quay.queue.items[0].current_work.name] == 'A':
                reward_prof_quay += 1

        w_1, w_2 = 1, 0
        reward = w_1 * reward_move_no + w_2 * reward_prof_quay
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

    done = False
    state = env.reset()
    r = []

    while not done:
        action = np.random.randint(20)

        next_state, reward, done = env.step(action)
        r.append(reward)
        state = next_state

        print(env.quays["Source"].sent)