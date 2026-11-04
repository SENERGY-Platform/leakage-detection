"""
   Copyright 2026 InfAI (CC SES)

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

import datetime
import tempfile
import unittest
from unittest import mock

import pandas as pd
from operator_lib.util.config import MissingConfigValueError
from operator_lib.util.helpers import HistoryNeedsRayError

from fake_timescale import Timescale
from main import CustomConfig, Operator

START = datetime.datetime(2026, 7, 1, 6, 0, 30, tzinfo=datetime.timezone.utc)


def readings(minutes, start=START):
    # a reading every minute of a meter whose consumption varies from minute to minute
    reading = 100.0
    out = []
    for minute in range(minutes):
        out.append((start + datetime.timedelta(minutes=minute), reading))
        reading += (1 + minute % 7) / 1000
    return out


class TestHistory(unittest.TestCase):
    def setUp(self):
        self.dirs = []
        self.timescale = Timescale()

    def tearDown(self):
        for directory in self.dirs:
            directory.cleanup()

    def start(self, reader=None, init_phase_length="60"):
        # an operator on a volume of its own that reads its history from self.timescale unless another reader is given
        directory = tempfile.TemporaryDirectory()
        self.dirs.append(directory)
        operator = Operator()
        operator.produce = lambda value: None
        operator.data_path = directory.name
        with reader or self.timescale.patched() as self.read:
            operator.init(None, None, None, None, None, None,
                          config=CustomConfig({"init_phase_length": init_phase_length,
                                               "init_phase_level": "d"}))
        return operator

    def live(self, operator, values):
        return [result for at, reading in values
                if (result := operator.run({"Consumption": reading}, "", "device", at)) is not None]

    def slots(self, operator):
        return {hour: entries for hour, entries in operator.time_window_consumption_list_dict.items() if entries}

    def test_reads_the_last_28_days_from_timescale(self):
        self.start()
        self.read.assert_called_once_with(pd.Timedelta(28, "d"))

    def test_derives_the_same_slots_from_the_history_as_from_live_readings(self):
        values = readings(180)
        live = self.start()
        self.live(live, values)
        for at, reading in values:
            self.timescale.write(at, reading)
        self.assertEqual(self.slots(live), self.slots(self.start()))

    def test_continues_the_slot_a_restart_cuts_through(self):
        values = readings(60)
        live = self.start()
        self.live(live, values)
        for at, reading in values[:37]:
            self.timescale.write(at, reading)
        restarted = self.start()
        self.live(restarted, values[37:])
        self.assertEqual(self.slots(live), self.slots(restarted))

    def test_leaves_out_the_readings_that_the_history_holds_already(self):
        values = readings(60)
        live = self.start()
        self.live(live, values)
        for at, reading in values[:40]:
            self.timescale.write(at, reading)
        restarted = self.start()
        # Kafka delivers again what came before the restart
        self.live(restarted, values[20:])
        self.assertEqual(self.slots(live), self.slots(restarted))

    def test_ends_the_initial_phase_at_once_with_a_history_as_long_as_it(self):
        values = readings(3 * 24 * 60)
        for at, reading in values[:-60]:
            self.timescale.write(at, reading)
        results = self.live(self.start(init_phase_length="2"), values[-60:])
        self.assertEqual("", results[0]["initial_phase"])
        self.assertTrue(all(result.get("time_window") for result in results[1:]))

    def test_counts_the_initial_phase_from_the_first_value_of_a_shorter_history(self):
        values = readings(24 * 60)
        for at, reading in values[:-60]:
            self.timescale.write(at, reading)
        results = self.live(self.start(init_phase_length="2"), values[-60:])
        self.assertEqual("Die Anwendung befindet sich noch bis ca. 2026-07-03T08:00:30 in der Initialisierungsphase",
                         results[0]["initial_phase"])

    def test_starts_without_history_when_only_kafka_holds_it(self):
        operator = self.start(reader=mock.patch("main.provide_historic_frames_local", side_effect=HistoryNeedsRayError("kafka")))
        self.assertEqual({}, self.slots(operator))

    def test_starts_without_history_without_a_database_connection(self):
        operator = self.start(reader=mock.patch("main.provide_historic_frames_local", side_effect=MissingConfigValueError("ts_conn")))
        self.assertEqual({}, self.slots(operator))

    def test_starts_without_history_when_the_consumption_comes_from_several_devices(self):
        frame = pd.DataFrame({"time": [pd.Timestamp("2026-07-01 06:00:30"), pd.Timestamp("2026-07-01 06:30:30")],
                              "Consumption": [100.0, 100.03], "device_id": ["a", "b"]})
        operator = self.start(reader=mock.patch("main.provide_historic_frames_local", return_value=[frame]))
        self.assertEqual({}, self.slots(operator))


if __name__ == '__main__':
    unittest.main()
