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
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import kneed
import os
from itertools import chain
import datetime
from collections import defaultdict

from operator_lib.util import Config
class CustomConfig(Config):
    data_path = "/opt/data"
    init_phase_length: float = 2
    init_phase_level: str = "d"

    def __init__(self, d, **kwargs):
        super().__init__(d, **kwargs)
        if self.init_phase_length != '':
            self.init_phase_length = float(self.init_phase_length)
        else:
            self.init_phase_length = 14
        
        if self.init_phase_level == '':
            self.init_phase_level = 'd'

class Operator(OperatorBase):
    configType = CustomConfig

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        self.data_path = self.config.data_path

        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

        self.first_data_time = load(self.config.data_path, "first_data_time.pickle")

        self.init_phase_duration = pd.Timedelta(self.config.init_phase_length, self.config.init_phase_level)

        self.init_phase_handler = InitPhase(self.data_path, self.init_phase_duration, self.first_data_time, self.produce)
        value = {
                    "value": 0,
                    "timestamp": "",
                    "message": "",
                    "last_consumptions": "",
                    "time_window": ""
        }
        self.init_phase_handler.send_first_init_msg(value)

        self.time_window_consumption_list_dict = defaultdict(list)
        self.time_window_consumption_list_dict_anomalies = defaultdict(list)

        self.window_boundaries_times =  [datetime.time(0, 0), datetime.time(1, 0), datetime.time(2, 0), datetime.time(3, 0), datetime.time(4, 0), datetime.time(5, 0), datetime.time(6, 0),
                                         datetime.time(7, 0), datetime.time(8, 0), datetime.time(9, 0), datetime.time(10, 0), datetime.time(11, 0), datetime.time(12, 0), datetime.time(13, 0), 
                                         datetime.time(14, 0), datetime.time(15, 0), datetime.time(16, 0), datetime.time(17, 0), datetime.time(18, 0), datetime.time(19, 0), datetime.time(20, 0),
                                         datetime.time(21, 0), datetime.time(22, 0), datetime.time(23, 0), datetime.time(23, 59, 59, 999000)]
                                      



        self.consumption_same_five_min = []

        self.time_window_consumption_clustering = {}

        self.time_window_consumption_list_dict = load(self.data_path, "time_window_consumption_list_dict.pickle", default=defaultdict(list))
        self.time_window_consumption_list_dict_anomalies = load(self.data_path, "time_window_consumption_list_dict_anomaly.pickle", default=defaultdict(list))

    def update_time_window_consumption_list_dict(self):
        min_index = np.argmin([float(datapoint['Consumption']) for datapoint in self.consumption_same_five_min])
        max_index = np.argmax([float(datapoint['Consumption']) for datapoint in self.consumption_same_five_min])
        five_min_consumption_max = float(self.consumption_same_five_min[max_index]['Consumption'])
        five_min_consumption_min = float(self.consumption_same_five_min[min_index]['Consumption'])
        overall_five_min_consumption = 1000*(five_min_consumption_max-five_min_consumption_min)
        if [time for time in self.window_boundaries_times if time<self.current_five_min.time()]==[]:
            self.last_time_window_start = self.window_boundaries_times[-2]
        else:
            self.last_time_window_start = max(time for time in self.window_boundaries_times if time<self.current_five_min.time())
        self.time_window_consumption_list_dict[str(self.last_time_window_start)].append((self.timestamp, overall_five_min_consumption))
        for i, entry in enumerate(self.time_window_consumption_list_dict[str(self.last_time_window_start)]):
            if self.timestamp-entry[0] > pd.Timedelta(28,'d'):
                del self.time_window_consumption_list_dict[str(self.last_time_window_start)][i]
        save(self.data_path, "time_window_consumption_list_dict.pickle", self.time_window_consumption_list_dict)
        return

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
        self.time_window_consumption_clustering[str(self.last_time_window_start)] = DBSCAN(eps=epsilon, min_samples=10).fit(np.array([five_min_consumption 
                                                                     for _, five_min_consumption in self.time_window_consumption_list_dict[str(self.last_time_window_start)]]).reshape(-1,1))
        return self.time_window_consumption_clustering[str(self.last_time_window_start)].labels_
    
    def test_time_window_consumption(self, clustering_labels):
        anomalous_indices = np.where(clustering_labels==-1)[0]
        quantile = np.quantile([five_min_consumption for _, five_min_consumption in self.time_window_consumption_list_dict[str(self.last_time_window_start)]],0.995)
        anomalous_indices_high = [i for i in anomalous_indices if self.time_window_consumption_list_dict[str(self.last_time_window_start)][i][1] > quantile]
        if len(self.time_window_consumption_list_dict[str(self.last_time_window_start)])-1 in anomalous_indices_high:
            logger.warning(f'In den letzten 5 Minuten wurde ungewöhnlich viel verbraucht.')
            self.time_window_consumption_list_dict_anomalies[str(self.last_time_window_start)].append(self.time_window_consumption_list_dict[str(self.last_time_window_start)][-1])
        save(self.data_path, "time_window_consumption_list_dict_anomaly.pickle", self.time_window_consumption_list_dict_anomalies)

        return [self.time_window_consumption_list_dict[str(self.last_time_window_start)][i] for i in anomalous_indices_high]
    
    def run(self, data, selector='energy_func', device_id=''):
        self.timestamp = todatetime(data['Time']).tz_localize(None)
        if not self.first_data_time:
            self.first_data_time = self.timestamp
            self.init_phase_handler = InitPhase(self.data_path, self.init_phase_duration, self.first_data_time, self.produce)
            save(self.data_path, "first_data_time.pickle", self.first_data_time)
        logger.debug('consumption: '+str(data['Consumption'])+'  '+'time: '+str(self.timestamp))

        operator_is_init = self.init_phase_handler.operator_is_in_init_phase(self.timestamp)
        
        init_value = {
                    "value": 0,
                    "timestamp": timestamp_to_str(self.timestamp),
                    "message": "",
                    "last_consumptions": "",
                    "time_window": ""
        }

        self.current_five_min = self.timestamp.floor('5T')
        if self.consumption_same_five_min == []:
            self.consumption_same_five_min.append(data)
            return
        elif self.consumption_same_five_min != []:
            if self.current_five_min==todatetime(self.consumption_same_five_min[-1]['Time']).tz_localize(None).floor('5T'):
                self.consumption_same_five_min.append(data)
                return
            else:
                self.update_time_window_consumption_list_dict()
                if not operator_is_init:
                    if self.init_phase_handler.init_phase_needs_to_be_reset():
                        self.consumption_same_five_min = [data] 
                        logger.debug(self.init_phase_handler.reset_init_phase(init_value))
                        return self.init_phase_handler.reset_init_phase(init_value)
                    epsilon = self.determine_epsilon()
                    clustering_labels = self.create_clustering(epsilon)
                    days_with_excessive_five_min_consumption_during_this_time_window_of_day = self.test_time_window_consumption(clustering_labels)
                    self.consumption_same_five_min = [data] 
                    df_cons_last_14_days = self.create_df_cons_last_14_days(days_with_excessive_five_min_consumption_during_this_time_window_of_day)              
                    if self.timestamp in list(chain.from_iterable(days_with_excessive_five_min_consumption_during_this_time_window_of_day)):
                        logger.debug(self.create_output(1, self.timestamp, df_cons_last_14_days))
                        return self.create_output(1, self.timestamp, df_cons_last_14_days) # Excessive time window consumption just detected.
                    else:
                        logger.debug(self.create_output(0, self.timestamp, df_cons_last_14_days))
                        return self.create_output(0, self.timestamp, df_cons_last_14_days) # No excessive consumption just detected.
                elif operator_is_init:
                    self.consumption_same_five_min = [data]
                    logger.debug(self.init_phase_handler.generate_init_msg(self.timestamp, init_value))
                    return self.init_phase_handler.generate_init_msg(self.timestamp, init_value)

        
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
        print(df.reset_index(inplace=False).to_json(orient="values"))
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

