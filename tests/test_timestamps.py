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


class TestOutputTimestamp(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.operator = Operator()
        self.operator.produce = lambda value: None
        self.operator.init(None, None, None, None, None, None,
                           config=CustomConfig({"data_path": self.dir.name, "init_phase_length": "7",
                                                "init_phase_level": "d"}))

    def tearDown(self):
        self.dir.cleanup()

    def results(self, start, minutes):
        out = []
        for minute in range(minutes):
            at = start + datetime.timedelta(minutes=minute)
            result = self.operator.run({"Consumption": 100 + minute * 0.001}, "", "device", at)
            if result is not None:
                out.append(result)
        return out

    def test_the_output_carries_the_utc_time_in_winter(self):
        start = datetime.datetime(2026, 10, 27, 8, 0, 30, tzinfo=datetime.timezone.utc)
        results = self.results(start, 7)
        self.assertEqual(["2026-10-27T08:05:30.000Z"], [r["timestamp"] for r in results])

    def test_the_output_carries_the_utc_time_in_summer(self):
        start = datetime.datetime(2026, 7, 1, 8, 0, 30, tzinfo=datetime.timezone.utc)
        results = self.results(start, 7)
        self.assertEqual(["2026-07-01T08:05:30.000Z"], [r["timestamp"] for r in results])


if __name__ == '__main__':
    unittest.main()
