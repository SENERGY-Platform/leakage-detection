"""
   Copyright 2022 InfAI (CC SES)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

__all__ = ("Operator", )

from operator_lib.util import OperatorBase, logger, InitPhase, todatetime, timestamp_to_str
from operator_lib.util.persistence import save, load
from operator_lib.util.config import MissingConfigValueError
from operator_lib.util.helpers import provide_historic_frames_local, HistoryNeedsRayError
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import kneed
import os
from itertools import chain
import datetime
from collections import defaultdict

import typing
import datetime

from operator_lib.util import Config, Inputs, Outputs

# The consumption is judged per slot of five minutes, against the slots of the same hour of the day in German time.
SLOT = pd.Timedelta(5, 'min')
TIMEZONE = 'Europe/Berlin'
# The slots an hour of the day needs before it is judged: the size of a cluster.
MIN_SLOTS = 10
# The slots of each hour of the day are kept this long.
HISTORY_LENGTH = pd.Timedelta(28, 'd')
# Between two readings further apart the course of the consumption is not known, so no slot is derived from them.
MAX_GAP = pd.Timedelta(2, 'h')


class CustomConfig(Config):
    init_phase_length: float = 2
    init_phase_level: str = "d"
    # A field of the library, declared again so that flows can set it.
    logger_level: str = "warning"

    def __init__(self, d, **kwargs):
        super().__init__(d, **kwargs)
        if self.init_phase_length != '':
            self.init_phase_length = float(self.init_phase_length)
        else:
            self.init_phase_length = 2
        
        if self.init_phase_level == '':
            self.init_phase_level = 'd'

# The time of a reading is the time of its Kafka message, so there is no input for it.
class LeakageInputs(Inputs):
    Consumption: float

class LeakageOutputs(Outputs):
    value: int
    timestamp: str
    message: str
    last_consumptions: str
    time_window: str
    initial_phase: str

class Operator(OperatorBase):
    configType = CustomConfig
    inputs = LeakageInputs
    outputs = LeakageOutputs
    # Where the start of the initial phase is kept, the volume of the deployment; not a config value, which a flow
    # could change.
    data_path = "/opt/data"

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

        self.time_window_consumption_clustering = {}

        # Per hour of the day: (end of the slot in German time, litres), oldest first. Like everything derived from
        # the meter readings not kept: it is rebuilt from the history at every start.
        self.time_window_consumption_list_dict = defaultdict(list)
        # The last meter reading and the start of the open slot with the reading there, each as (UTC time, reading).
        self.last_reading = None
        self.slot_start = None
        history = self.read_history()
        for at, reading in history:
            self.add_reading(at, reading)

        self.first_data_time = load(self.data_path, "first_data_time.pickle")
        if history:
            first = history[0][0].tz_convert(TIMEZONE).tz_localize(None)
            if not self.first_data_time or first < self.first_data_time:
                # The initial phase counts from the first value of the history, so a history of its length ends it at once.
                self.first_data_time = first
                save(self.data_path, "first_data_time.pickle", self.first_data_time)

        self.init_phase_duration = pd.Timedelta(self.config.init_phase_length, self.config.init_phase_level)

        # No message before the first value: without a time of its own the export drops it, and the end of the
        # initial phase is not known yet. The first judged slot announces the initial phase.
        self.init_phase_handler = InitPhase(self.data_path, self.init_phase_duration, self.first_data_time, self.produce)

    def read_history(self):
        # Timescale is the only source of the history: it is read at every start and extended live, so a restart or a
        # new volume loses nothing. Any other failure of the read fails the start, and the platform starts it again.
        try:
            frames = provide_historic_frames_local(HISTORY_LENGTH)
        except (HistoryNeedsRayError, MissingConfigValueError) as e:
            logger.warning(f'Starting without history: {e}')
            return []
        if len(frames) != 1 or 'device_id' in frames[0].columns:
            logger.warning('Starting without history: the consumption comes from more than one input or device')
            return []
        frame = frames[0].dropna(subset=['Consumption']).sort_values('time', kind='stable')
        return list(zip(pd.to_datetime(frame['time'], utc=True), frame['Consumption'].astype(float)))

    def add_reading(self, at, reading):
        # Adds the meter reading at the UTC time `at` and returns the slots it completes, as (hour, entry). The
        # reading at a slot boundary is interpolated linearly between the readings around it: so a slot gets the
        # consumption of its own five minutes, whatever the rate the meter reports at, and none is lost.
        completed = []
        if self.last_reading is not None:
            last_at, last_value = self.last_reading
            if at <= last_at:
                # A repeated or late reading: the slots up to the last reading are derived already.
                return completed
            if at - last_at > MAX_GAP or reading < last_value:
                # Across a long gap the course of the consumption is not known, and a falling reading means a new
                # meter: the open slot is left out.
                self.slot_start = None
            else:
                boundary = last_at.floor(SLOT) + SLOT
                while boundary <= at:
                    value = last_value + (reading - last_value) * ((boundary - last_at) / (at - last_at))
                    if self.slot_start is not None:
                        start, start_value = self.slot_start
                        completed.append(self.add_slot(start, boundary, 1000 * (value - start_value)))
                    self.slot_start = (boundary, value)
                    boundary += SLOT
        if self.slot_start is None and at == at.floor(SLOT):
            self.slot_start = (at, reading)
        self.last_reading = (at, reading)
        return completed

    def add_slot(self, start, end, litres):
        # The slot belongs to the hour of the day in German time it starts in. Slots older than the history length
        # are dropped from that hour; the times grow, so they are the first ones.
        hour = datetime.time(start.tz_convert(TIMEZONE).hour)
        entry = (end.tz_convert(TIMEZONE).tz_localize(None), litres)
        entries = self.time_window_consumption_list_dict[str(hour)]
        entries.append(entry)
        while entry[0] - entries[0][0] > HISTORY_LENGTH:
            entries.pop(0)
        return hour, entry

    def determine_epsilon(self):
        neighbors = NearestNeighbors(n_neighbors=5)
        neighbors_fit = neighbors.fit(np.array([five_min_consumption for _, five_min_consumption in self.time_window_consumption_list_dict[str(self.last_time_window_start)]]).reshape(-1,1))
        distances, _ = neighbors_fit.kneighbors(np.array([five_min_consumption for _, five_min_consumption in self.time_window_consumption_list_dict[str(self.last_time_window_start)]]).reshape(-1,1))
        distances = np.sort(distances, axis=0)
        distances_x = distances[:,1]
        kneedle = kneed.KneeLocator(np.linspace(0,1,len(distances_x)), distances_x, S=0.9, curve="convex", direction="increasing")
        epsilon = kneedle.knee_y
        if epsilon==0 or epsilon==None:
            return 1
        else:
            return epsilon

    def create_clustering(self, epsilon):
        self.time_window_consumption_clustering[str(self.last_time_window_start)] = DBSCAN(eps=epsilon, min_samples=MIN_SLOTS).fit(np.array([five_min_consumption 
                                                                     for _, five_min_consumption in self.time_window_consumption_list_dict[str(self.last_time_window_start)]]).reshape(-1,1))
        return self.time_window_consumption_clustering[str(self.last_time_window_start)].labels_
    
    def test_time_window_consumption(self, clustering_labels):
        anomalous_indices = np.where(clustering_labels==-1)[0]
        quantile = np.quantile([five_min_consumption for _, five_min_consumption in self.time_window_consumption_list_dict[str(self.last_time_window_start)]],0.995)
        anomalous_indices_high = [i for i in anomalous_indices if self.time_window_consumption_list_dict[str(self.last_time_window_start)][i][1] > quantile]
        if len(self.time_window_consumption_list_dict[str(self.last_time_window_start)])-1 in anomalous_indices_high:
            logger.warning(f'In den letzten 5 Minuten wurde ungewöhnlich viel verbraucht.')

        return [self.time_window_consumption_list_dict[str(self.last_time_window_start)][i] for i in anomalous_indices_high]
    
    def run(self, data: typing.Dict[str, typing.Any], selector: str, device_id, timestamp: datetime.datetime):
        # Convert to german time and then forget the timezone. The time windows are of the local
        # day; the output carries the UTC time, since the export reads it as UTC.
        utc = pd.Timestamp(timestamp).tz_convert("UTC")
        self.timestamp = utc.tz_convert(TIMEZONE).tz_localize(None)

        if not self.first_data_time:
            self.first_data_time = self.timestamp
            self.init_phase_handler = InitPhase(self.data_path, self.init_phase_duration, self.first_data_time, self.produce)
            save(self.data_path, "first_data_time.pickle", self.first_data_time)
        logger.debug('consumption: '+str(data['Consumption'])+'  '+'time: '+str(self.timestamp))

        operator_is_init = self.init_phase_handler.operator_is_in_init_phase(self.timestamp)
        
        init_value = {
                    "value": 0,
                    "timestamp": timestamp_to_str(utc),
                    "message": "",
                    "last_consumptions": "",
                    "time_window": ""
        }

        completed = self.add_reading(utc, float(data['Consumption']))
        if not completed:
            return
        if operator_is_init:
            output = self.init_phase_handler.generate_init_msg(self.timestamp, init_value)
        elif self.init_phase_handler.init_phase_needs_to_be_reset():
            output = self.init_phase_handler.reset_init_phase(init_value)
        else:
            # A reading of a meter that reports seldom can complete several slots, even of two hours. They are all
            # judged; the output shows the hour of the last one and reports any excessive one.
            anomaly = 0
            for hour in dict.fromkeys(hour for hour, _ in completed):
                self.last_time_window_start = hour
                days_with_excessive_five_min_consumption_during_this_time_window_of_day = self.judge_time_window()
                if any(entry in days_with_excessive_five_min_consumption_during_this_time_window_of_day
                       for slot_hour, entry in completed if slot_hour == hour):
                    anomaly = 1
            df_cons_last_14_days = self.create_df_cons_last_14_days(days_with_excessive_five_min_consumption_during_this_time_window_of_day)
            output = self.create_output(anomaly, utc, df_cons_last_14_days)
        logger.debug(output)
        return output

    def judge_time_window(self):
        # The excessive slots of the hour self.last_time_window_start.
        if len(self.time_window_consumption_list_dict[str(self.last_time_window_start)]) < MIN_SLOTS:
            # Below the size of a cluster every slot is an outlier, and below five the neighbours cannot even be
            # determined.
            return []
        epsilon = self.determine_epsilon()
        clustering_labels = self.create_clustering(epsilon)
        return self.test_time_window_consumption(clustering_labels)

    def create_df_cons_last_14_days(self, days_with_excessive_five_min_consumption_during_this_time_window_of_day):
        ends_of_5min_slots_in_time_window = [timestamp for timestamp, _ in self.time_window_consumption_list_dict[str(self.last_time_window_start)] if 
                                             self.timestamp-timestamp < pd.Timedelta(14, "d")]
        time_window_5min_consumptions = [consumption for timestamp, consumption in self.time_window_consumption_list_dict[str(self.last_time_window_start)] if 
                                         self.timestamp-timestamp < pd.Timedelta(14, "d")]
        anomalies_check_list = []
        for timestamp, _ in self.time_window_consumption_list_dict[str(self.last_time_window_start)]:
            if self.timestamp-timestamp < pd.Timedelta(14, "d"):
                if timestamp in list(chain.from_iterable(days_with_excessive_five_min_consumption_during_this_time_window_of_day)):
                    anomalies_check_list.append(1)
                else:
                    anomalies_check_list.append(0)    
        df = pd.DataFrame({0:time_window_5min_consumptions, 1:anomalies_check_list}, index=ends_of_5min_slots_in_time_window)
        return df.reset_index(inplace=False).to_json(orient="values")
    
    def create_output(self, anomaly, timestamp, df_cons_last_14_days):
        if anomaly == 0:
            return {
                    "value": anomaly,
                    "timestamp": timestamp_to_str(timestamp),
                    "last_consumptions": df_cons_last_14_days,
                    "time_window": f'{str(self.last_time_window_start)}-{str((datetime.datetime.combine(datetime.date.today(), self.last_time_window_start) +datetime.timedelta(hours=1)).time())}'
            }
        elif anomaly == 1:
            message = f'In den letzten 5 Minuten wurde übermäßig viel Wasser verbraucht.'
        return {
                    "value": anomaly,
                    "timestamp": timestamp_to_str(timestamp),
                    "message": message,
                    "last_consumptions": df_cons_last_14_days,
                    "time_window": f'{str(self.last_time_window_start)}-{str((datetime.datetime.combine(datetime.date.today(), self.last_time_window_start) +datetime.timedelta(hours=1)).time())}'
        }    


from operator_lib.operator_lib import OperatorLib
if __name__ == "__main__":
    OperatorLib(Operator(), name="leakage-detection-operator", git_info_file='git_commit')

