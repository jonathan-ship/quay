import os
import simpy
import pandas as pd


class Work:
    def __init__(self, name, start, finish, cut, duration_fix, duration):
        self.name = name
        self.start = start
        self.finish = finish
        self.cut = cut
        self.duration_fix = duration_fix
        self.duration = duration

        self.working_time = self.finish - self.start + 1
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


class Quay:
    def __init__(self, env, model, name, length, score_table):
        self.env = env
        self.model = model
        self.name = name
        self.length = length
        self.scores = score_table.to_dict()

        self.queue = simpy.Store(env)
        self.occupied = False
        self.working_start = 0.0

        self.action = self.env.process(self._run())

    def _run(self):
        while True:
            ship = yield self.queue.get()
            ship.current_work = ship.work_list[ship.fix_idx]
            ship.current_work.quay = self.name
            self.occupied = True

            if ship.current_work.cut == "N":
                working_time = ship.current_work.duration
            else:
                working_time = ship.current_work.duration_fix

            try:
                yield self.env.timeout(working_time)
                ship.current_work.progress += working_time
            except simpy.Interrupt:
                ship.current_work.progress += (self.env.now - self.working_start)


class Source:
    def __init__(self, env, ships, model, monitor):
        self.env = env
        self.name = 'Source'
        self.ships = ships
        self.model = model
        self.monitor = monitor

        self.action = env.process(self._run())

    def _run(self):
        while True:
            ship = self.ships.pop(0)

            IAT = ship.current_work.start - self.env.now
            if IAT > 0:
                yield self.env.timeout(ship.current_work.working_time - self.env.now)

            if ship.fix_idx == 0:
                self.monitor.record(self.env.now, "launching", self.name, ship.name, None)

            quay = ship.current_work.quay
            self.model[quay].put(ship)

            if len(self.ships) == 0:
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


