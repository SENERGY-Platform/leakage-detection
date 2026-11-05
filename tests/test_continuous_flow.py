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

from fake_timescale import Timescale
from main import CustomConfig, Operator

START = datetime.datetime(2026, 7, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
SLOT = datetime.timedelta(minutes=5)
# what a slot of an even flow consumes, in litres
EVEN = 0.5
MESSAGE = ("Seit mindestens 24 Stunden fließt ununterbrochen Wasser: "
           "Der Zähler stand in dieser Zeit nie 60 Minuten still.")


class TestContinuousFlow(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.timescale = Timescale()
        self.reading = 100.0

    def tearDown(self):
        self.dir.cleanup()

    def start(self, **config):
        # without an initial phase, so that every slot is judged
        operator = Operator()
        operator.produce = lambda value: None
        operator.data_path = self.dir.name
        with self.timescale.patched():
            operator.init(None, None, None, None, None, None,
                          config=CustomConfig({"init_phase_length": "0", "init_phase_level": "d", **config}))
        return operator

    def write_history(self, slots):
        # readings at the ends of slots of an even flow up to START
        for slot in range(-slots, 1):
            self.timescale.write(START + slot * SLOT, self.reading)
            self.reading += EVEN / 1000
        self.reading -= EVEN / 1000

    def alarms(self, operator, litres_per_slot):
        # readings at the ends of the slots from START on, none for a slot of None; (slot, message) of each alarm
        operator.run({"Consumption": self.reading}, "", "device", START)
        alarms = []
        for slot, litres in enumerate(litres_per_slot):
            if litres is None:
                continue
            self.reading += litres / 1000
            result = operator.run({"Consumption": self.reading}, "", "device", START + (slot + 1) * SLOT)
            if result is not None and result.get("value") == 1:
                alarms.append((slot, result["message"]))
        return alarms

    def test_reports_a_meter_that_has_not_stood_still_for_24_hours(self):
        self.assertEqual([(287, MESSAGE)], self.alarms(self.start(), [EVEN] * 300))

    def test_reports_the_continuous_flow_once_until_the_meter_stands_still(self):
        self.assertEqual([287], [slot for slot, _ in self.alarms(self.start(), [EVEN] * 600)])

    def test_reports_again_after_the_meter_stood_still_for_an_hour(self):
        alarms = self.alarms(self.start(), [EVEN] * 300 + [0.0] * 12 + [EVEN] * 300)
        self.assertEqual([287, 312 + 287], [slot for slot, _ in alarms])

    def test_does_not_count_a_standstill_shorter_than_an_hour(self):
        alarms = self.alarms(self.start(), [EVEN] * 150 + [0.0] * 11 + [EVEN] * 150)
        self.assertEqual([287], [slot for slot, _ in alarms])

    def test_takes_the_standstill_from_the_config(self):
        alarms = self.alarms(self.start(rest_minutes="30"), [EVEN] * 150 + [0.0] * 6 + [EVEN] * 200)
        self.assertEqual([], alarms)

    def test_counts_a_gap_of_more_than_two_hours_as_standing_still(self):
        # whether the meter stood still in the gap is not known, and an outage raises no alarm
        alarms = self.alarms(self.start(), [EVEN] * 150 + [None] * 36 + [EVEN] * 150)
        self.assertEqual([], alarms)

    def test_reports_no_continuous_flow_with_0_hours(self):
        self.assertEqual([], self.alarms(self.start(continuous_flow_hours="0"), [EVEN] * 300))

    def test_counts_the_time_without_standstill_from_the_history(self):
        self.write_history(23 * 12)
        alarms = self.alarms(self.start(), [EVEN] * 24)
        self.assertEqual([11], [slot for slot, _ in alarms])


if __name__ == '__main__':
    unittest.main()
