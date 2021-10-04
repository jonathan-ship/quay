import random

import simpy
import numpy as np

from environment.simulation import *


class QuayScheduling:
    def __init__(self, df_quay, df_score, df_weight, df_ship, df_work, df_work_fixed, log_dir, duration=365, frame=5):
        self.df_quay = df_quay
        self.df_score = df_score
        self.df_weight = df_weight
        self.df_ship = df_ship
        self.df_work = df_work
        self.df_work_fixed = df_work_fixed
        self.log_dir = log_dir
        self.duration = duration
        self.frame = frame

        self.state_size = len(self.df_weight["그룹"].unique()) * frame + len(self.df_quay) * 3 + 3
        self.action_size = len(self.df_quay) + 1

        self.e = 0
        self.total_score = 0.0
        self.max_score = sum([max(df_score.loc[(row["선종"], row["작업명"])]) for i, row in self.df_work.iterrows()
                          if row["작업명"] != "시운전" and row["작업명"] != "G/T"])
        self.cnt_day = 0
        self.cnt_total = 0
        self.max_cnt_total = sum([df_weight["가중치"][row["선종"]] for i, row in self.df_work.iterrows()])
        self.move_constraint = 5
        self.time = 0.0
        self.w_move = 0.5
        self.w_efficiency = 0.5
        self.reward_move = 0.0
        self.reward_efficiency = 0.0
        self.mapping = {i: row["안벽"] for i, row in self.df_quay.iterrows()}
        self.mapping[len(self.df_quay)] = "S"
        self.inverse_mapping = {y: x for x, y in self.mapping.items()}
        self.sim_env, self.ships, self.model, self.monitor = self._modeling()

    def step(self, action):
        done = False
        # Take action at current decision time step
        quay_name = self.mapping[action]
        ship_category = self.model["Routing"].ship.category
        work_category = self.model["Routing"].ship.current_work.name
        if self.model["Routing"].decision == None:
            print("f")
        self.model["Routing"].decision.succeed(quay_name)
        self.model["Routing"].indicator = False
        # if self.model["Routing"].ship.name == "PROJ_44":
        #     print(self.sim_env.now, self.mapping[action], self.model["Routing"].ship.current_work.name)

        if self.model["Routing"].current_quay != quay_name:
            self.cnt_total += self.df_weight["가중치"][ship_category]
            self.cnt_day += 1
        if quay_name != "S":
            self.total_score += self.model[quay_name].scores[ship_category, work_category]

        # Run until next decision time step
        while True:
            # Check whether there is any decision time step
            if self.model["Routing"].indicator:
                if self.sim_env.now != self.time:
                    self.cnt_day = 0
                    self.time = self.sim_env.now
                break

            if self.model["Sink"].ships_rec == len(self.df_ship):
                done = True
                self.sim_env.run()
                self.model["Sink"].monitor.save_event_tracer()
                break

            self.sim_env.step()

        reward = self._calculate_reward()
        next_state = self._get_state()

        # if self.cnt_day == self.move_constraint:
        #     reward = -100
        #     done = True
        #     self.model["Sink"].monitor.save_event_tracer()

        return next_state, reward, done

    def reset(self):
        self.e += 1
        self.sim_env, self.ships, self.model, self.monitor = self._modeling()
        self.done = False
        self.move = 0

        # Go to first decision time step
        while True:
            # Check whether there is any decision time step
            if self.model["Routing"].indicator:
                break
            self.sim_env.step()

        return self._get_state()

    def _get_state(self):
        f_0 = np.zeros((len(self.df_weight["그룹"].unique()), self.frame))  # 진수될 선박에 대한 정보
        f_1 = np.zeros(len(self.df_quay))  # 의사결정 시점에서 해당 작업의 각 안벽에 대한 선호도
        f_2 = np.zeros(len(self.df_quay))  # 안벽 이동 가능 여부 (0.0, 0.25, 0.5, 0.75, 1.0)
        f_3 = np.zeros(len(self.df_quay)) # 해당 안벽에 작업되고 있는 작업의 해당 안벽에서의 선호도
        f_4 = np.zeros(1) # 하루 누적 이동횟수
        f_5 = np.zeros(2) # 시뮬레이션 히스토리

        for i, row in self.df_ship.iterrows():
            if self.sim_env.now <= row["진수일"] < self.sim_env.now + self.duration:
                row_idx = int(self.df_weight.loc[row["선종"], ["그룹"]])
                col_idx = int((row["진수일"] - self.sim_env.now) * self.frame / self.duration)
                f_0[row_idx, col_idx] += 1

        decision_work_name = self.model["Routing"].ship.current_work.name  # 해당 안벽에서 작업중인 작업이름
        decision_category = self.model["Routing"].ship.category

        for idx, quay_name in self.mapping.items():
            if quay_name not in ["Routing", "Source", "Sink", "S"]:
                f_1[idx] = self.model[quay_name].scores[decision_category, decision_work_name]

                if self.model["Routing"].possible_quay.get(quay_name):
                    if self.model["Routing"].possible_quay[quay_name] == 1:
                        if self.df_weight["가중치"][self.model[quay_name].ship.category] == 3:
                            f_2[idx] = 0.75
                        elif self.df_weight["가중치"][self.model[quay_name].ship.category] == 4:
                            f_2[idx] = 0.5
                        elif self.df_weight["가중치"][self.model[quay_name].ship.category] == 6:
                            f_2[idx] = 0.25
                    elif self.model["Routing"].possible_quay[quay_name] == 2:
                        f_2[idx] = 1

                if self.model[quay_name].occupied and int(self.model[quay_name].back) == 0:
                    category = self.model[quay_name].ship.category  # 해당 안벽에서 작업중인 선박의 선종
                    work_name = self.model[quay_name].ship.current_work.name  # 해당 안벽에서 작업중인 작업이름
                    f_3[idx] = self.model[quay_name].scores[category, work_name]

        f_4[0] = self.cnt_day / self.move_constraint
        f_5[0] = self.cnt_total / self.max_cnt_total
        f_5[1] = self.total_score / self.max_score

        state = np.concatenate((f_0.flatten(), f_1, f_2, f_3, f_4, f_5), axis=0)
        return state

    def _calculate_reward(self):
        ship_category = self.model["Routing"].move["ship_category"]
        work_category = self.model["Routing"].move["work_category"]
        previous_quay = self.model["Routing"].move["previous"]
        current_quay = self.model["Routing"].move["current"]
        loss = self.model["Routing"].move["loss"]

        # 호선이동 횟수 역수할지 - 둘지 고민
        reward_move = 0
        if loss:
            reward_move = 1 / 7
        else:
            if previous_quay != current_quay:
                if previous_quay != "Source":
                    reward_move = 1 / self.df_weight["가중치"][ship_category]
            else:
                reward_move = 1
        # 전문 안벽 배치율
        reward_eff = 0
        if current_quay != "S":
            reward_eff = self.model[current_quay].scores[ship_category, work_category] / 1.5

        self.reward_move = reward_move
        self.reward_efficiency = reward_eff
        reward = self.w_move * reward_move + self.w_efficiency * reward_eff

        return reward

    def _modeling(self):
        sim_env = simpy.Environment()  # 시뮬레이션 환경 생성
        monitor = Monitor(self.log_dir + 'log_%d.csv' % self.e)  # 시뮬레이션 이벤트 로그 기록을 위한 Monitor 객체 생성

        # 안벽 배치 대상 선박에 대한 리스트 생성
        ships = []
        for i, row in self.df_ship.iterrows():
            works = self.df_work[self.df_work["호선번호"] == row["호선번호"]]  # 선박의 선종에 해당하는 안벽 작업 데이터프레임 선택
            # 선박의 작업 중 안벽 고정 작업이 있는 지 확인
            if row["호선번호"] in self.df_work_fixed["호선번호"]:
                # 고정 작업이 있을 경우
                fix_name = self.df_work_fixed[self.df_work_fixed["호선번호"] == row["호선번호"]]["작업명"]  # 고정 작업명
                fix_idx = int(works[works["작업명"] == fix_name]["순번"]) - 1  # 선박의 작업 리스트 중 고정 작업의 순서 index
            else:
                # 고정 작업이 없을 경우
                fix_idx = 0

            # Ship 클래스에 대한 객체 생성 후 선박 리스트에 추가
            ship = Ship(row["호선번호"], row["선종"], row["길이"], row["진수일"], row["인도일"], works, fix_idx=fix_idx)
            ships.append(ship)

        # 시뮬레이션 프로세스 모델링
        model = {}
        model["Source"] = Source(sim_env, ships, model, monitor)  # Source
        for i, row in self.df_quay.iterrows():
            scores = self.df_score[row["안벽"]]
            shared_quay_set = []
            for j in range(1, 4):
                if row["공유{0}".format(j)]:
                    shared_quay_set.append(row["공유{0}".format(j)])
            quay = Quay(sim_env, row["안벽"], model, row["길이"], shared_quay_set, scores, monitor)
            model[row["안벽"]] = quay  # Quay
        model["S"] = Sea(sim_env, model, monitor)  # Sea
        model["Routing"] = Routing(sim_env, model, monitor)  # Routing
        model["Sink"] = Sink(sim_env, model, monitor)  # Sink

        return sim_env, ships, model, monitor


if __name__ == "__main__":

    from data import *

    info_path = "./data/기준정보.xlsx"
    scenario_path = "./data/호선정보_학습.xlsx"

    log_dir = '../result/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    df_quay, df_score, df_weight, df_ship, df_work, df_work_fix, start = import_train_data(info_path, scenario_path)
    env = QuayScheduling(df_quay, df_score, df_weight, df_ship, df_work, df_work_fix, log_dir)

    done = False
    state = env.reset()
    r = []

    while not done:
        possible_action = env.model["Routing"].possible_quay
        possible_action = [env.inverse_mapping[key] for key in possible_action.keys()]
        action = random.choice(possible_action)

        next_state, reward, done = env.step(action)
        r.append(reward)
        state = next_state

        print(reward)
        print(next_state)

    export_result(log_dir + "log_%d.csv" % 1, start)