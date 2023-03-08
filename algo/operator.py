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

import util
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import kneed
import os
from itertools import chain
import pickle
import datetime
from collections import defaultdict

class Operator(util.OperatorBase):
    def __init__(self, device_id, data_path):
        if not os.path.exists(data_path):
            os.mkdir(data_path)

        self.time_window_consumption_list_dict = defaultdict(list)
        self.time_window_consumption_list_dict_anomalies = defaultdict(list)
        self.data_history = pd.Series([], index=[],dtype=object)

        self.window_boundaries_times =  [datetime.time(0, 0), datetime.time(1, 0), datetime.time(2, 0), datetime.time(3, 0), datetime.time(4, 0), datetime.time(5, 0), datetime.time(6, 0),
                                         datetime.time(7, 0), datetime.time(8, 0), datetime.time(9, 0), datetime.time(10, 0), datetime.time(11, 0), datetime.time(12, 0), datetime.time(13, 0), 
                                         datetime.time(14, 0), datetime.time(15, 0), datetime.time(16, 0), datetime.time(17, 0), datetime.time(18, 0), datetime.time(19, 0), datetime.time(20, 0),
                                         datetime.time(21, 0), datetime.time(22, 0), datetime.time(23, 0), datetime.time(23, 59, 59, 999000)]
                                      



        self.consumption_same_five_min = []

        self.current_time_window_start = None
        self.timestamp = None
        self.last_time_window_start = None

        self.time_window_consumption_clustering = {}

        self.clustering_file_path = f'{data_path}/clustering.pickle'
        self.epsilon_file_path = f'{data_path}/epsilon.pickle'
        self.time_window_consumption_list_dict_file_path = f'{data_path}/time_window_consumption_list_dict.pickle'
        self.time_window_consumption_list_dict_anomaly_file_path = f'{data_path}/time_window_consumption_list_dict_anomaly.pickle'

    def todatetime(self, timestamp):
        if str(timestamp).isdigit():
            if len(str(timestamp))==13:
                return pd.to_datetime(int(timestamp), unit='ms')
            elif len(str(timestamp))==19:
                return pd.to_datetime(int(timestamp), unit='ns')
        else:
            return pd.to_datetime(timestamp)

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
        with open(self.time_window_consumption_list_dict_file_path, 'wb') as f:
            pickle.dump(self.time_window_consumption_list_dict, f)
        return

    def determine_epsilon(self):
        neighbors = NearestNeighbors(n_neighbors=5)
        neighbors_fit = neighbors.fit(np.array([five_min_consumption for _, five_min_consumption in self.time_window_consumption_list_dict[str(self.last_time_window_start)]]).reshape(-1,1))
        distances, _ = neighbors_fit.kneighbors(np.array([five_min_consumption for _, five_min_consumption in self.time_window_consumption_list_dict[str(self.last_time_window_start)]]).reshape(-1,1))
        distances = np.sort(distances, axis=0)
        distances_x = distances[:,1]
        kneedle = kneed.KneeLocator(np.linspace(0,1,len(distances_x)), distances_x, S=0.9, curve="convex", direction="increasing")
        epsilon = kneedle.knee_y
        with open(self.epsilon_file_path, 'wb') as f:
            pickle.dump(epsilon, f)
        if epsilon==0 or epsilon==None:
            return 1
        else:
            return epsilon

    def create_clustering(self, epsilon):
        self.time_window_consumption_clustering[str(self.last_time_window_start)] = DBSCAN(eps=epsilon, min_samples=10).fit(np.array([five_min_consumption 
                                                                     for _, five_min_consumption in self.time_window_consumption_list_dict[str(self.last_time_window_start)]]).reshape(-1,1))
        with open(self.clustering_file_path, 'wb') as f:
            pickle.dump(self.time_window_consumption_clustering, f)
        return self.time_window_consumption_clustering[str(self.last_time_window_start)].labels_
    
    def test_time_window_consumption(self, clustering_labels):
        anomalous_indices = np.where(clustering_labels==-1)[0]
        quantile = np.quantile([five_min_consumption for _, five_min_consumption in self.time_window_consumption_list_dict[str(self.last_time_window_start)]],0.995)
        anomalous_indices_high = [i for i in anomalous_indices if self.time_window_consumption_list_dict[str(self.last_time_window_start)][i][1] > quantile]
        if len(self.time_window_consumption_list_dict[str(self.last_time_window_start)])-1 in anomalous_indices_high:
            print(f'In den letzten 5 Minuten wurde ungewöhnlich viel verbraucht.')
            self.time_window_consumption_list_dict_anomalies[str(self.last_time_window_start)].append(self.time_window_consumption_list_dict[str(self.last_time_window_start)][-1])
        with open(self.time_window_consumption_list_dict_anomaly_file_path, 'wb') as f:
            pickle.dump(self.time_window_consumption_list_dict_anomalies,f)

        return [self.time_window_consumption_list_dict[str(self.last_time_window_start)][i] for i in anomalous_indices_high]
    
    def run(self, data, selector='energy_func'):
        self.timestamp = self.todatetime(data['Time']).tz_localize(None)
        print('energy: '+str(data['Consumption'])+'  '+'time: '+str(self.timestamp))
        self.data_history = pd.concat([self.data_history, pd.Series([float(data['Consumption'])], index=[self.timestamp])])
        self.current_five_min = self.timestamp.floor('5T')
        if self.consumption_same_five_min == []:
            self.consumption_same_five_min.append(data)
            return
        elif self.consumption_same_five_min != []:
            if self.current_five_min==self.todatetime(self.consumption_same_five_min[-1]['Time']).tz_localize(None).floor('5T'):
                self.consumption_same_five_min.append(data)
                return
            else:
                self.update_time_window_consumption_list_dict()
                if len(self.time_window_consumption_list_dict[str(self.last_time_window_start)]) >= 5:
                    epsilon = self.determine_epsilon()
                    clustering_labels = self.create_clustering(epsilon)
                    days_with_excessive_five_min_consumption_during_this_time_window_of_day = self.test_time_window_consumption(clustering_labels)
                    self.consumption_same_five_min = [data]                 
                    if self.timestamp in list(chain.from_iterable(days_with_excessive_five_min_consumption_during_this_time_window_of_day)):
                        return {'value': f'Nachricht vom {str(self.timestamp.date())} um {str(self.timestamp.hour)}:{str(self.timestamp.minute)} Uhr: In den letzten 5 Minuten wurde übermäßig viel Wasser verbraucht.'} # Excessive time window consumption just detected.
                    else:
                        return  # No excessive consumtion just detected.
                else:
                    self.consumption_same_five_min = [data]
                    return
