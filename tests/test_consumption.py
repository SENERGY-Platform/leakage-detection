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

from main import CustomConfig, Operator


def at(day_time, day="2026-07-01"):
    return datetime.datetime.fromisoformat(f"{day}T{day_time}+00:00")


def meter(start, end, minutes, litres_per_minute, first_reading=100.0):
    # readings every few minutes from start to end, both included, of a meter that runs evenly
    readings = []
    time = start
    while time <= end:
        readings.append((time, first_reading + litres_per_minute * (time - start).total_seconds() / 60 / 1000))
        time += datetime.timedelta(minutes=minutes)
    return readings


class TestConsumption(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.dir.cleanup()

    def slots(self, readings):
        # the slots kept per hour of the day, by their end in German time; in the initial phase, so nothing is judged
        operator = Operator()
        operator.produce = lambda value: None
        operator.init(None, None, None, None, None, None,
                      config=CustomConfig({"data_path": self.dir.name, "init_phase_length": "60",
                                           "init_phase_level": "d"}))
        for time, reading in readings:
            operator.run({"Consumption": reading}, "", "device", time)
        return {hour: [(end.strftime("%Y-%m-%d %H:%M"), round(litres, 6)) for end, litres in entries]
                for hour, entries in operator.time_window_consumption_list_dict.items() if entries}

    def test_counts_the_whole_consumption_of_a_meter_that_reports_every_minute(self):
        slots = self.slots(meter(at("06:00:30"), at("06:30:30"), 1, 1.0))
        self.assertEqual({"08:00:00": [("2026-07-01 08:10", 5.0), ("2026-07-01 08:15", 5.0), ("2026-07-01 08:20", 5.0),
                                       ("2026-07-01 08:25", 5.0), ("2026-07-01 08:30", 5.0)]}, slots)

    def test_counts_the_whole_consumption_of_a_meter_that_reports_every_15_minutes(self):
        slots = self.slots(meter(at("06:00:00"), at("07:00:00"), 15, 1.0))
        self.assertEqual({"08:00:00": [(f"2026-07-01 {end}", 5.0) for end in
                                       ("08:05", "08:10", "08:15", "08:20", "08:25", "08:30",
                                        "08:35", "08:40", "08:45", "08:50", "08:55", "09:00")]}, slots)

    def test_shares_the_consumption_between_two_readings_out_by_time(self):
        slots = self.slots([(at("06:02:00"), 100.0), (at("06:12:00"), 100.01), (at("06:16:00"), 100.01)])
        # 3 of the 10 litres before 06:05, 5 in the slot to 06:10, then 2 until the meter stands still
        self.assertEqual({"08:00:00": [("2026-07-01 08:10", 5.0), ("2026-07-01 08:15", 2.0)]}, slots)

    def test_files_a_slot_under_the_hour_of_german_time_it_starts_in(self):
        slots = self.slots(meter(at("06:50:30"), at("07:05:30"), 1, 1.0))
        self.assertEqual({"08:00:00": [("2026-07-01 09:00", 5.0)], "09:00:00": [("2026-07-01 09:05", 5.0)]}, slots)

    def test_files_the_slots_a_reading_after_a_gap_completes_under_the_hours_they_start_in(self):
        readings = meter(at("06:30:30"), at("06:36:30"), 1, 1.0)
        slots = self.slots(readings + [(at("08:01:30"), readings[0][1] + 0.091)])
        self.assertEqual([f"2026-07-01 08:{minute}" for minute in ("40", "45", "50", "55")] + ["2026-07-01 09:00"],
                         [end for end, _ in slots["08:00:00"]])
        self.assertEqual(12, len(slots["09:00:00"]))
        self.assertEqual({5.0}, {litres for entries in slots.values() for _, litres in entries})

    def test_leaves_out_the_slots_across_a_gap_of_more_than_two_hours(self):
        before = meter(at("06:00:30"), at("06:10:30"), 1, 1.0)
        after = meter(at("08:30:30"), at("08:40:30"), 1, 1.0, first_reading=before[-1][1] + 0.14)
        slots = self.slots(before + after)
        self.assertEqual({"08:00:00": [("2026-07-01 08:10", 5.0)], "10:00:00": [("2026-07-01 10:40", 5.0)]}, slots)

    def test_leaves_out_the_slot_in_which_the_meter_reading_falls(self):
        old = meter(at("06:00:30"), at("06:11:30"), 1, 1.0)
        new = meter(at("06:12:30"), at("06:21:30"), 1, 1.0, first_reading=0.0)
        slots = self.slots(old + new)
        self.assertEqual({"08:00:00": [("2026-07-01 08:10", 5.0), ("2026-07-01 08:20", 5.0)]}, slots)

    def test_ignores_a_reading_that_is_not_newer_than_the_last(self):
        slots = self.slots([(at("06:04:30"), 100.0), (at("06:05:30"), 100.001), (at("06:05:30"), 100.001),
                            (at("06:03:30"), 100.5), (at("06:10:30"), 100.006)])
        self.assertEqual({"08:00:00": [("2026-07-01 08:10", 5.0)]}, slots)

    def test_keeps_the_slots_of_the_last_28_days(self):
        slots = self.slots(meter(at("00:00:00", day="2026-06-01"), at("23:00:00", day="2026-06-30"), 60, 1.0))
        for hour, entries in slots.items():
            oldest, newest = (datetime.datetime.strptime(entries[i][0], "%Y-%m-%d %H:%M") for i in (0, -1))
            self.assertEqual(datetime.timedelta(days=28), newest - oldest, hour)

    def test_files_the_hour_that_repeats_when_the_clocks_go_back_under_that_hour_twice(self):
        # 02:00 to 03:00 German time passes twice on 2026-10-25
        slots = self.slots(meter(at("23:58:30", day="2026-10-24"), at("02:02:30", day="2026-10-25"), 1, 1.0))
        self.assertEqual(["02:00:00"], list(slots))
        self.assertEqual([5.0] * 24, [litres for _, litres in slots["02:00:00"]])


if __name__ == '__main__':
    unittest.main()
