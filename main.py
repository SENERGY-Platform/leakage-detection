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
import os
import datetime
from collections import Counter, defaultdict

import typing
import datetime

from operator_lib.util import Config, Inputs, Outputs

# The consumption is judged per slot of five minutes, against the slots of the same hour of the day in German time.
SLOT = pd.Timedelta(5, 'min')
TIMEZONE = 'Europe/Berlin'
# The slots of each hour of the day are kept this long.
HISTORY_LENGTH = pd.Timedelta(28, 'd')
# Between two readings further apart the course of the consumption is not known, so no slot is derived from them.
MAX_GAP = pd.Timedelta(2, 'h')
# A slot is excessive above the alarm level of its hour. The tail of the slots of an hour begins above this quantile of
# them; a generalized Pareto distribution is fitted to it, and the level is the consumption that it exceeds with the
# probability that gives the false alarms per month of the config.
TAIL_QUANTILE = 0.9
# An hour with fewer slots in its tail is judged with the tail of the whole day; while the whole day has fewer, no slot
# is judged.
MIN_TAIL = 10
# The shape of the tail is taken no lower than 0, an exponential tail: estimated from a few dozen slots it comes out
# below 0 by chance, which caps the level just above the largest slot seen and lets the next larger shower raise an
# alarm. Towards 0.5 its estimate no longer holds.
MAX_SHAPE = 0.45
# What the false alarms per month refer to.
SLOTS_PER_MONTH = pd.Timedelta(30, 'd') / SLOT
# The slots are compared to 10 ml, far below what a water meter resolves: slots of an even flow differ by the rounding
# of floating point numbers only, which would make a tail of a few picolitres and an alarm of every larger slot.
DECIMALS = 2


def alarm_level(litres, probability):
    # The consumption of a slot that the slots `litres` exceed with `probability`, or None if too few of them lie in
    # their tail.
    litres = np.round(np.asarray(litres, dtype=float), DECIMALS)
    if len(litres) == 0:
        return None
    threshold = np.quantile(litres, TAIL_QUANTILE)
    excesses = litres[litres > threshold] - threshold
    if len(excesses) < MIN_TAIL:
        return None
    shape = tail_shape(excesses)
    # the mean of the excesses is scale / (1 - shape)
    scale = excesses.mean() * (1 - shape)
    # the probability within the tail
    ratio = probability * len(litres) / len(excesses)
    if shape == 0:
        return float(threshold - scale * np.log(ratio))
    return float(threshold + scale / shape * (ratio ** -shape - 1))


def tail_shape(excesses):
    # The shape of a generalized Pareto distribution from the probability weighted moments of the excesses over its
    # threshold (Hosking and Wallis, 1987), between 0 and MAX_SHAPE.
    excesses = np.sort(excesses)
    n = len(excesses)
    a0 = excesses.mean()
    a1 = np.sum((n - np.arange(1, n + 1)) / (n - 1) * excesses) / n
    if a0 - 2 * a1 <= 0:
        # excesses that are all alike: as light a tail as can be
        return 0.0
    return float(np.clip(2 - a0 / (a0 - 2 * a1), 0, MAX_SHAPE))


def config_number(value, default):
    # A flow that sets no value sends an empty string.
    return default if value == '' else float(value)


class CustomConfig(Config):
    init_phase_length: float = 2
    init_phase_level: str = "d"
    false_alarms_per_month: float = 1
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

        self.false_alarms_per_month = config_number(self.false_alarms_per_month, 1)
        if not self.false_alarms_per_month > 0:
            raise ValueError(f'false_alarms_per_month must be above 0, not {self.false_alarms_per_month}')

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

        # The probability with which a slot of usual consumption lies above the alarm level of its hour.
        self.false_alarm_probability = self.config.false_alarms_per_month / SLOTS_PER_MONTH

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

    def alarm_levels(self, completed):
        # The alarm level of each hour of the completed slots, from the slots before them: those of the hour, or of the
        # whole day where the hour has too few in its tail; None while neither has enough. The completed slots are the
        # last ones of their hours, and a meter that reports seldom completes several of the same consumption, which are
        # judged alike.
        new = Counter(str(hour) for hour, _ in completed)
        before = {hour: [litres for _, litres in entries[:len(entries) - new[hour]]]
                  for hour, entries in self.time_window_consumption_list_dict.items()}
        levels = {hour: alarm_level(before[hour], self.false_alarm_probability) for hour in new}
        if None in levels.values():
            whole_day = alarm_level([litres for slots in before.values() for litres in slots],
                                    self.false_alarm_probability)
            levels = {hour: whole_day if level is None else level for hour, level in levels.items()}
        return levels

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
            levels = self.alarm_levels(completed)
            messages = []
            if any(levels[str(hour)] is not None and litres > levels[str(hour)] for hour, (_, litres) in completed):
                messages.append('In den letzten 5 Minuten wurde übermäßig viel Wasser verbraucht.')
            for message in messages:
                logger.warning(message)
            self.last_time_window_start = completed[-1][0]
            df_cons_last_14_days = self.create_df_cons_last_14_days(levels[str(self.last_time_window_start)])
            output = self.create_output(messages, utc, df_cons_last_14_days)
        logger.debug(output)
        return output

    def create_df_cons_last_14_days(self, level):
        # The slots of the last 14 days in the hour self.last_time_window_start, each with 1 if it lies above the alarm
        # level of the hour.
        entries = [(end, litres) for end, litres in self.time_window_consumption_list_dict[str(self.last_time_window_start)]
                   if self.timestamp - end < pd.Timedelta(14, "d")]
        df = pd.DataFrame({0: [litres for _, litres in entries],
                           1: [int(level is not None and litres > level) for _, litres in entries]},
                          index=[end for end, _ in entries])
        return df.reset_index(inplace=False).to_json(orient="values")

    def create_output(self, messages, timestamp, df_cons_last_14_days):
        output = {
                "value": int(bool(messages)),
                "timestamp": timestamp_to_str(timestamp),
        }
        if messages:
            output["message"] = ' '.join(messages)
        output["last_consumptions"] = df_cons_last_14_days
        output["time_window"] = f'{str(self.last_time_window_start)}-{str((datetime.datetime.combine(datetime.date.today(), self.last_time_window_start) +datetime.timedelta(hours=1)).time())}'
        return output


from operator_lib.operator_lib import OperatorLib
if __name__ == "__main__":
    OperatorLib(Operator(), name="leakage-detection-operator", git_info_file='git_commit')

