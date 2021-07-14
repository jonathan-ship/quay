import os
import simpy
import pandas as pd


class Work:
    def __init__(self, name, start, finish, cut, duration_fix, duration):
        self.name = name  # 작업의 이름
        self.start = start  # 작업의 시작일
        self.finish = finish  # 작업의 종료일
        self.cut = cut  # 자르기 종류(N, S, F)
        self.duration_fix = duration_fix  # 필수 기간
        self.duration = duration

        self.working_time = self.finish - self.start + 1 # 전체 작업 기간(작업의 종료일 - 작업의 시작일 + 1)
        self.done = False  # 작업 종료 여부
        self.progress = 0.0  # 작업 진행 정도
        self.quay = None  # 작업이 수행되는 안벽


class Ship:
    def __init__(self, name, category, length, launching_date, delivery_date, work_table, fix_idx=0):
        self.name = name  # 호선 번호
        self.category = category  # 선종
        self.length = length  # 선박의 길이
        self.launching_date = launching_date  # 선박의 진수일 (L/C)
        self.delivery_date = delivery_date  # 선박의 인도일 (D/L)
        self.work_table = work_table  # 선박에 대한 안벽 작업 리스트 (데이터 프레임 형태)
        self.fix_idx = fix_idx  # 안벽 작업들 중 이미 안벽이 고정된 작업을 제외하고 처음으로 고려되는 작업에 대한 인덱스

        self.total_duration = self.delivery_date - self.launching_date + 1  # 모든 안벽 작업을 완료하기 위해 필요한 작업 기간
        # 선박에 대한 안벽 작업 리스트 (Work 클래스로 구성된 리스트 형태)
        # 데이터프레임 형태의 안벽 작업 리스트를 바탕으로 Work 클래스의 객체로 구성된 안벽 작업 리스트를 생성
        self.work_list = [Work(row["작업"], row["착수(%)"], row["종료(%)"], row["자르기"], row["필수기간"],
                               self.total_duration * (int(row["종료(%)"] - int(row["착수(%)"]))) / 100)
                          for i, row in work_table.iterrows()]
        self.current_work = self.work_list[self.fix_idx]  # 현재 수행되고 있는 작업


class Quay:
    def __init__(self, env, name, model, length, score_table, monitor):
        self.env = env  # simpy 시뮬레이션 환경
        self.name = name  # 안벽 번호
        self.model = model  # 전체 안벽에 대한 정보(모든 Quay 클래스의 객체를 갖고 있는 딕셔너리)
        self.length = length  # 안벽의 길이
        self.scores = score_table.to_dict()  # 해당 안벽에 대한 각 안벽 작업들의 점수
        self.monitor = monitor  # 이벤트의 기록을 위한 Monitor 클래스의 객체

        self.queue = simpy.Store(env)  # 안벽에서 작업을 수행할 선박을 넣어주는 simpy Store 객체
        self.ship_in = None  # 현재 안벽에 배치된 선박
        self.ship_out = None  # 다른 안벽으로 이동해야 하는 선박
        self.waiting = False  # 의사 결정을 대기하는 이벤트의 존재 여부
        self.decision = None  # 강화학습 에이전트의 의사 결정 시점인 decision point를 알려주는 simpy event 객체
        self.occupied = False  # 안벽의 점유 여부
        self.cut_possible = False  # 현재 안벽에서 수행되는 작업에 대한 자르기 가능 여부
        self.working_start = 0.0  # 현재 안벽 작업의 시작 시간

        self.action = self.env.process(self.run())

    def run(self):
        while True:
            self.ship_in = yield self.queue.get()  # 현재 안벽에서 작업을 수행할 선박
            self.occupied = True  # 안벽을 점유 상태로 설정

            if self.ship_in.current_work.cut == "N":
                # 자르기 불가(N)
                # 안벽에서의 작업 시간은 해당 작업의 전체 작업으로 설정됨
                working_time = self.ship_in.current_work.duration
            else:
                # 자르기 가능(S, F)
                if self.ship_in.current_work.progress < self.ship_in.current_work.duration_fix:
                    # 처음 작업이 시작된 경우, 안벽에서의 작업 시간은 필수 기간에 해당하는 시간으로 설정됨
                    working_time = self.ship_in.current_work.duration_fix
                else:
                    # 이미 작업이 수행되었으나 자르기 후 다시 시작하는 경우, 안벽에서의 작업 시간은 남은 작업 시간으로 설정됨
                    # 즉, 필수 기간이 끝난 시점에만 안벽 이동 여부를 결정하고 그 이후에는 종료시까지 특정 안벽에서 계속 작업 수행
                    working_time = self.ship_in.current_work.working_time - self.ship_in.current_work.progress
                    self.cut_possible = True

            try:
                # 앞서 결정된 작업 시간에 해당하는 시간 동안 작업 수행
                yield self.env.timeout(working_time)
                self.ship_in.current_work.progress += working_time
            except simpy.Interrupt:
                # 다른 안벽(ex. A 안벽)에서 작업 완료된 선박에 의해 현재 안벽(ex. B 안벽)의 작업에 대한 자르기가 이루어진 경우
                # 현재 안벽(ex. B 안벽)에서 선박의 작업을 중단
                self.ship_in.current_work.progress += (self.env.now - self.working_start)
                self.ship_out = self.ship_in
            else:
                # 작업이 중간에 자르기 없이 완료된 경우
                self.ship_in.current_work.done = True  # 작업의 완료
                self.ship_in.fix_idx += 1  # 다음 작업의 인덱스
                self.ship_in.current_work = self.ship_in.work_list[self.ship_in.fix_idx]  # 다음으로 수행할 작업을 현재 작업으로 변경

            # 현재 안벽에 배치된 선박을 이동
            self.env.process(self.move(self.ship_in))

            self.occupied = False
            self.cut_possible = False

    def move(self, ship):
        determined = False
        while not determined:
            self.waiting = True
            self.decision = self.env.event()  # decision point에 도달했음을 알리는 이벤트 생성
            quay_name, determined = yield self.decision  # 해당 이벤트를 발생시키고 에이전트로부터 이동할 안벽 변호를 받음
            self.decision = None
            self.model[quay_name].queue.put(ship)  # 입력 받은 안벽으로 선박을 이동

            # 이동할 안벽이 정해졌을 때, 이동할 안벽에 다른 작업이 수행 중이면 해당 작업을 중단
            if quay_name and self.model[quay_name].occupied:
                self.model[quay_name].action.interrupt()

        self.ship_out = None


class Sea:
    def __init__(self, env, model, monitor):
        self.env = env  # simpy 시뮬레이션 환경
        self.name = "S"
        self.model = model  # 전체 안벽에 대한 정보(모든 Quay 클래스의 객체를 갖고 있는 딕셔너리)
        self.monitor = monitor  # 이벤트의 기록을 위한 Monitor 클래스의 객체

        self.queue = simpy.Store(env)  # 해상 작업을 수행할 선박을 넣어주는 simpy Store 객체
        self.waiting = {}  # 의사 결정을 대기하는 이벤트의 존재 여부
        self.decision = {}  # 강화학습 에이전트의 의사 결정 시점인 decision point를 알려주는 simpy event 객체의 딕셔너리
        self.ship_in = {}  # 해상에 있는 선박을 담는 딕셔너리

        self.action = self.env.process(self.run())

    def run(self):
        while True:
            ship = yield self.queue.get()  # 해상 작업을 수행할 선박
            self.ship_in[ship.name] = ship  # 해당 선박을 선박 딕셔너리에 등록
            self.env.process(self.sub_run(self.ship_in[ship.name]))

    def sub_run(self, ship):
        # 선박이 해상 작업을 마치고 안벽에 재배치될 때까지 반복
        while self.ship_in.get(ship.name):
            if ship.current_work.done:
                # 배치 안벽이 없어서 해상에 정박 중인 경우
                # 일마다 배치 가능 안벽 결정
                working_time = 1
            else:
                # 해상 작업(EX. 시운전)을 수행해야 하는 경우, 작업 시간은 해당 해상 작업의 작업 기간으로 설정
                working_time = ship.current_work.working_time

            # 앞서 결정된 작업 시간에 해당하는 시간 동안 작업 수행 또는 하루 동안 대기
            yield self.env.timeout(working_time)
            # 해상 작업이 완료된 경우
            if not ship.current_work.done:
                ship.current_work.done = True  # 작업의 완료
                ship.fix_idx += 1  # 다음 작업의 인덱스
                ship.current_work = ship.work_list[ship.fix_idx]  # 다음으로 수행할 작업을 현재 작업으로 변경

            determined = False
            while not determined:
                self.waiting[ship.name] = True
                self.decision[ship.name] = self.env.event()  # decision point에 도달했음을 알리는 이벤트 생성
                quay_name, determined = yield self.decision[ship.name]  # 해당 이벤트를 발생시키고 에이전트로부터 이동할 안벽 변호를 받음
                del self.decision[ship.name]
                self.model[quay_name].queue.put(ship)  # 입력 받은 안벽으로 선박을 이동

                # 이동할 안벽이 정해졌을 때, 이동할 안벽에 다른 작업이 수행 중이면 해당 작업을 중단
                if quay_name and self.model[quay_name].occupied:
                    self.model[quay_name].action.interrupt()

            del self.ship_in[ship.name]


class Source:
    def __init__(self, env, ships, model, monitor):
        self.env = env  # simpy 시뮬레이션 환경
        self.name = 'Source'
        self.ships = ships  # 전체 선박 리스트
        self.model = model  # 전체 안벽에 대한 정보(모든 Quay 클래스의 객체를 갖고 있는 딕셔너리)
        self.monitor = monitor  # 이벤트의 기록을 위한 Monitor 클래스의 객체

        self.sent = 0  # 진수(L/C)된 선박 수
        self.waiting = False  # 의사 결정을 대기하는 이벤트의 존재 여부
        self.decision = None  # 강화학습 에이전트의 의사 결정 시점인 decision point를 알려주는 simpy event 객체
        self.action = env.process(self.run())
        self.waiting_ships = []     # 의사결정 대기중인 선박

    def run(self):
        while True:
            ship = self.ships[self.sent]  # 진수 선박

            IAT = ship.current_work.start - self.env.now  # 현재 시점부터 선박의 진수까지 남은 기간
            # 남은 기간 동안 대기
            if IAT > 0:
                yield self.env.timeout(ship.current_work.working_time - self.env.now)

            if ship.fix_idx == 0:
                self.monitor.record(self.env.now, "launching", self.name, ship.name, None)

            determined = False
            while not determined:
                self.waiting = True
                self.waiting_ships.append(ship)
                self.decision = self.env.event()  # decision point에 도달했음을 알리는 이벤트 생성
                quay_name, determined = yield self.decision  # 해당 이벤트를 발생시키고 에이전트로부터 이동할 안벽 변호를 받음
                self.decision = None
                self.model[quay_name].queue.put(ship)  # 입력 받은 안벽으로 선박을 이동

                # 이동할 안벽이 정해졌을 때, 이동할 안벽에 다른 작업이 수행 중이면 해당 작업을 중단
                if quay_name and self.model[quay_name].occupied:
                    self.model[quay_name].action.interrupt()

            self.sent += 1

            # 모든 선박이 진수된 경우 시뮬레이션 종료
            if self.sent == len(self.ships):
                print("All ships are sent.")
                break


class Sink:
    def __init__(self, env, model, monitor):
        self.env = env  # simpy 시뮬레이션 환경
        self.name = 'Sink'
        self.model = model  # 전체 안벽에 대한 정보(모든 Quay 클래스의 객체를 갖고 있는 딕셔너리)
        self.monitor = monitor  # 이벤트의 기록을 위한 Monitor 클래스의 객체

        self.ships_rec = 0  # 인도(D/L)된 선박 수
        self.last_delivery = 0.0  # 가장 마지막 선박이 인도된 시점

    def put(self, ship):
        self.ships_rec += 1
        self.last_delivery = self.env.now


class Monitor(object):
    def __init__(self, filepath):
        self.filepath = filepath  # 이벤트 로그 파일 생성 경로

        self.time = []  # 이벤트 발생 시각
        self.event = []  # 발생된 이벤트 종류
        self.quay_name = []  # 이벤트가 발생된 프로세스(Quay, Sea, Source, Sink)
        self.ship_name = []  # 발생된 이벤트와 관련된 선박의 호선 번호
        self.work_name = []  # 발생된 이벤트와 관련된 작업

    def record(self, time=0.0, event="launching", quay_name="A1", ship_name="PROJ_01", work_name="화물창작업"):
        self.time.append(time)
        self.event.append(event)
        self.quay_name.append(quay_name)
        self.ship_name.append(ship_name)
        self.work_name.append(work_name)

    def save_event_tracer(self):
        event_tracer = pd.DataFrame(columns=['Time', 'Event', 'Quay', 'Ship', 'Work'])
        event_tracer['Time'] = self.time
        event_tracer['Event'] = self.event
        event_tracer['Quay'] = self.quay_name
        event_tracer['Ship'] = self.ship_name
        event_tracer['Work'] = self.work_name
        event_tracer.to_csv(self.filepath)

        return event_tracer


if __name__ == "__main__":
    from data import import_data

    info_path = "./data/기준정보+위탁과제용.xlsx"
    scenario_path = "./data/[수정] 호선일정정보+위탁과제.xlsx"

    df_quay, df_work, df_score, df_ship, df_work_fix = import_data(info_path, scenario_path)

    ships = []
    for i, row in df_ship.iterrows():
        works = df_work[df_work["선종"] == row["선종"]]
        if row["호선번호"] in df_work_fix["호선번호"]:
            fix_name = df_work_fix[df_work_fix["호선번호"] == row["호선번호"]]["작업명"]
            fix_idx = int(works[works["작업"] == fix_name]["순번"]) - 1
        else:
            fix_idx = 0

        ship = Ship(row["호선번호"], row["선종"], row["길이"], row["진수일"], row["인도일"], works, fix_idx=fix_idx)
        ships.append(ship)

    env = simpy.Environment()
    filepath = '../result/log.csv'
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    monitor = Monitor(filepath)

    quays = {}
    quays["Source"] = Source(env, ships, quays, monitor)
    for i, row in df_quay.iterrows():
        scores = df_score[row["안벽"]]
        quay = Quay(env, row["안벽"], quays, row["길이"], scores, monitor)
        quays[row["안벽"]] = quay
    quays["S"] = Sea(env, quays, monitor)
    quays["Sink"] = Sink(env, quays, monitor)


