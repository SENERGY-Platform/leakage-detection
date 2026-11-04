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

START = datetime.datetime(2026, 7, 1, 8, 0, 30, tzinfo=datetime.timezone.utc)


class TestJudgement(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.dir.cleanup()

    def start(self):
        # without an initial phase, so that every slot of the first hour is judged
        operator = Operator()
        operator.produce = lambda value: None
        operator.init(None, None, None, None, None, None,
                      config=CustomConfig({"data_path": self.dir.name, "init_phase_length": "0",
                                           "init_phase_level": "d"}))
        return operator

    def judgements(self, litres_per_slot):
        # a reading every minute; each 5-minute slot consumes its litres evenly
        operator = self.start()
        results = []
        reading = 100.0
        for slot, litres in enumerate(litres_per_slot):
            for minute in range(5):
                result = operator.run({"Consumption": reading}, "", "device",
                                      START + datetime.timedelta(minutes=5 * slot + minute))
                if result is not None and result.get("time_window"):
                    results.append(result["value"])
                reading += litres / 5 / 1000
        return results

    def test_raises_no_alarm_while_the_hour_holds_fewer_than_ten_slots(self):
        results = self.judgements([1.0, 1.2, 1.1, 0.9, 1.0, 1.2, 1.1, 50.0, 1.0, 1.0])
        self.assertTrue(results, "no slot was judged")
        self.assertEqual([0] * len(results), results)

    def test_raises_an_alarm_for_a_slot_far_above_the_others_of_its_hour(self):
        results = self.judgements([1.0, 1.2, 1.1, 0.9, 1.0, 1.2, 1.1, 0.9, 1.0, 1.2, 1.1, 50.0, 1.0])
        self.assertEqual(1, results[-1])


if __name__ == '__main__':
    unittest.main()
