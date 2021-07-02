import os
import simpy
import pandas as pd

decision_point = False
target = None


class Work:
    def __init__(self, name, start, finish, cut, duration_fix, duration):
        self.name = name
        self.start = start
        self.finish = finish
        self.cut = cut
        self.duration_fix = duration_fix
        self.duration = duration

        self.working_time = self.finish - self.start + 1
        self.progress = 0.0
        self.quay = None


class Ship:
    def __init__(self, name, category, length, launching_date, delivery_date, work_table, fix_idx=0):
        self.name = name
        self.category = category
        self.length = length
        self.launching_date = launching_date
        self.delivery_date = delivery_date
        self.work_table = work_table
        self.fix_idx = fix_idx

        self.total_duration = self.delivery_date - self.launching_date + 1
        self.work_list = [Work(row["작업"], row["착수(%)"], row["종료(%)"], row["자르기"], row["필수기간"],
                               self.total_duration * (int(row["종료(%)"] - int(row["착수(%)"]))) / 100)
                          for i, row in work_table.iterrows()]
        self.current_work = self.work_list[self.fix_idx]
        self.stopped = False


class Quay:
    def __init__(self, env, name, model, length, score_table):
        self.env = env
        self.name = name
        self.model = model
        self.length = length
        self.scores = score_table.to_dict()

        self.queue = simpy.Store(env)
        self.decision = None
        self.occupied = False
        self.working_start = 0.0

        self.action = self.env.process(self._run())

    def _run(self):
        while True:
            ship = yield self.queue.get()
            self.occupied = True

            if ship.current_work.cut == "N":
                working_time = ship.current_work.duration
            else:
                if ship.current_work.progress < ship.current_work.duration_fix:
                    working_time = ship.current_work.duration_fix
                else:
                    working_time = ship.current_work.working_time - ship.current_work.progress

            try:
                yield self.env.timeout(working_time)
                ship.current_work.progress += working_time
            except simpy.Interrupt as i:
                ship.current_work.progress += (self.env.now - self.working_start)
                quay_name = i.cause
            else:
                self.decision = self.env.event()
                quay_name = yield self.decision
                self.decision = None

            self.occupied = False
            if not ship.stopped:
                ship.fix_idx += 1
                ship.current_work = ship.work_list[ship.fix_idx]

            self.model[quay_name].queue.put(ship)

            if self.model[quay_name].occupied:
                self.model[quay_name].action.interrupt(self.name)


class Sea:
    def __init__(self, env, model):
        self.env = env
        self.name = "S"
        self.model = model
        self.queue = simpy.Store(env)

        self.ship_in_sea = {}
        self.action = self.env.process(self._run())

    def _run(self):
        while True:
            self.ship_in_sea[] = yield self.queue.get()
            if ship.current_work.name == "시운전":
                self.env.process(self.sea_trial(ship))

    def sea_trial(self, ship):
        yield self.env.timeout(ship.current_work.working_time)
        while


class Source:
    def __init__(self, env, ships, model, monitor):
        self.env = env
        self.name = 'Source'
        self.ships = ships
        self.model = model
        self.monitor = monitor

        self.sent = 0
        self.decision = None
        self.action = env.process(self._run())

    def _run(self):
        while True:
            ship = self.ships[self.sent]

            IAT = ship.current_work.start - self.env.now
            if IAT > 0:
                yield self.env.timeout(ship.current_work.working_time - self.env.now)

            if ship.fix_idx == 0:
                self.monitor.record(self.env.now, "launching", self.name, ship.name, None)

            self.decision = self.env.event()
            quay_name = yield self.decision
            self.decision = None
            self.model[quay_name].queue.put(ship)

            self.sent += 1

            if self.sent == len(self.ships):
                print("All ships are sent.")
                break


class Sink:
    def __init__(self, env, model, monitor):
        self.env = env
        self.name = 'Sink'
        self.model = model
        self.monitor = monitor

        self.ships_rec = 0
        self.last_delivery = 0.0

    def put(self, ship):
        self.ships_rec += 1
        self.last_delivery = self.env.now


class Monitor(object):
    def __init__(self, filepath):
        self.filepath = filepath

        self.time = []
        self.event = []
        self.quay_name = []
        self.ship_name = []
        self.work_name = []

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
        quay = Quay(row["안벽"], row["길이"], scores)
        quays[row["안벽"]] = quay
    quays["Sink"] = Sink(env, quays, monitor)

    print(":D")


