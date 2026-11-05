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

import contextlib
import datetime
import io
import json
import random
import tempfile
import unittest

from fake_timescale import Timescale
from main import CustomConfig, Operator

# 08:00 German time; the history ends here and the live readings begin
LIVE = datetime.datetime(2026, 7, 15, 6, 0, 0, tzinfo=datetime.timezone.utc)
SLOT = datetime.timedelta(minutes=5)


def household(litres_in_hour, end=LIVE, days=14):
    # readings every five minutes over the days before end; a slot consumes what litres_in_hour returns for the hour
    # of the day in German summer time and a random number generator
    rng = random.Random(1)
    readings = []
    reading = 100.0
    at = end - datetime.timedelta(days=days)
    while at <= end:
        readings.append((at, reading))
        reading += litres_in_hour((at + datetime.timedelta(hours=2)).hour, rng) / 1000
        at += SLOT
    return readings


def small_uses(hour, rng):
    return 0.0 if rng.random() < 0.6 else rng.uniform(1, 10)


class TestJudgement(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.timescale = Timescale()
        self.reading = 100.0

    def tearDown(self):
        self.dir.cleanup()

    def start(self, history=()):
        # without an initial phase, so that every slot is judged, and without the continuous flow, which the slots of
        # these households, consuming independently of each other, never stop
        for at, reading in history:
            self.timescale.write(at, reading)
            self.reading = reading
        operator = Operator()
        operator.produce = lambda value: None
        operator.data_path = self.dir.name
        with self.timescale.patched():
            operator.init(None, None, None, None, None, None,
                          config=CustomConfig({"init_phase_length": "0",
                                               "init_phase_level": "d",
                                               "continuous_flow_hours": "0"}))
        return operator

    def judge(self, operator, litres_per_slot):
        # a reading at the end of each slot from LIVE on, after a first slot of nothing whose output ends the initial
        # phase; the outputs that judge a slot
        results = []
        for slot, litres in enumerate([0.0] + litres_per_slot):
            self.reading += litres / 1000
            result = operator.run({"Consumption": self.reading}, "", "device", LIVE + (slot + 1) * SLOT)
            if result is not None and result.get("time_window"):
                results.append(result)
        return results

    def test_raises_an_alarm_for_a_slot_far_above_the_others_of_its_hour(self):
        results = self.judge(self.start(household(small_uses)), [3.0, 5.0, 200.0])
        self.assertEqual([0, 0, 1], [result["value"] for result in results])
        self.assertEqual("In den letzten 5 Minuten wurde übermäßig viel Wasser verbraucht.", results[-1]["message"])

    def test_raises_no_alarm_for_slots_like_the_others_of_their_hour(self):
        results = self.judge(self.start(household(small_uses)), [3.0, 5.0, 9.0])
        self.assertEqual([0, 0, 0], [result["value"] for result in results])

    def test_raises_no_alarm_for_a_slot_just_above_the_largest_of_its_hour(self):
        # the hour has never consumed more than 10 litres a slot
        results = self.judge(self.start(household(small_uses)), [10.5])
        self.assertEqual([0], [result["value"] for result in results])

    def test_marks_the_excessive_slot_in_the_last_consumptions(self):
        results = self.judge(self.start(household(small_uses)), [3.0, 5.0, 200.0])
        slots = json.loads(results[-1]["last_consumptions"])
        self.assertEqual([200.0], [round(litres, 6) for _, litres, excessive in slots if excessive == 1])

    def test_raises_no_alarm_while_the_whole_day_holds_fewer_than_ten_slots_in_its_tail(self):
        results = self.judge(self.start(), [float(slot % 7) for slot in range(30)] + [1000.0])
        self.assertTrue(results, "no slot was judged")
        self.assertEqual([0] * len(results), [result["value"] for result in results])

    def test_judges_an_hour_with_too_few_slots_in_its_tail_against_the_whole_day(self):
        # nothing is ever consumed at 8 o'clock, up to 40 litres a slot at other hours
        quiet_at_8 = lambda hour, rng: 0.0 if hour == 8 or rng.random() < 0.5 else rng.uniform(5, 40)
        results = self.judge(self.start(household(quiet_at_8)), [30.0])
        self.assertEqual([0], [result["value"] for result in results])

    def test_raises_an_alarm_in_an_hour_with_too_few_slots_in_its_tail_for_a_slot_far_above_the_whole_day(self):
        quiet_at_8 = lambda hour, rng: 0.0 if hour == 8 or rng.random() < 0.5 else rng.uniform(5, 40)
        results = self.judge(self.start(household(quiet_at_8)), [500.0])
        self.assertEqual([1], [result["value"] for result in results])

    def test_writes_nothing_to_standard_output(self):
        # what is worth seeing goes through the logger, whose level the config sets
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.judge(self.start(household(small_uses)), [3.0, 5.0, 200.0])
        self.assertEqual("", output.getvalue())

    def test_raises_an_alarm_for_an_excessive_slot_of_the_hour_before_the_one_a_reading_ends_in(self):
        # about 5 litres a slot, at 11 o'clock also showers; from 10:00 German time a reading every minute until the
        # meter pauses after 10:56:30, and its next reading completes the last slot of that hour, which holds a good
        # part of the 60 litres in between, and the first one of the next, which holds more but not more than a shower
        start = LIVE + datetime.timedelta(hours=2)
        showers_at_11 = lambda hour, rng: rng.uniform(20, 60) if hour == 11 and rng.random() < 0.5 else rng.uniform(4, 6)
        operator = self.start(household(showers_at_11, end=start))
        readings = [(start + datetime.timedelta(minutes=minute, seconds=30),
                     self.reading + (minute + 0.1 * (minute % 3)) / 1000) for minute in range(57)]
        readings.append((start + datetime.timedelta(minutes=66, seconds=30), readings[-1][1] + 0.06))
        results = [result for at, reading in readings
                   if (result := operator.run({"Consumption": reading}, "", "device", at)) is not None]
        self.assertEqual([0] * (len(results) - 1), [result["value"] for result in results[:-1]])
        self.assertEqual(("11:00:00-12:00:00", 1), (results[-1]["time_window"], results[-1]["value"]))


if __name__ == '__main__':
    unittest.main()
