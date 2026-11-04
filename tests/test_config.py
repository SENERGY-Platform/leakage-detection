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

import unittest

from main import CustomConfig


class TestConfig(unittest.TestCase):
    def initial_phase(self, values):
        config = CustomConfig(values)
        return config.init_phase_length, config.init_phase_level

    def test_takes_an_initial_phase_of_two_days_when_its_length_is_missing(self):
        self.assertEqual((2.0, "d"), self.initial_phase({}))

    def test_takes_an_initial_phase_of_two_days_when_its_length_is_empty(self):
        self.assertEqual((2.0, "d"), self.initial_phase({"init_phase_length": "", "init_phase_level": ""}))

    def test_takes_the_initial_phase_that_is_set(self):
        self.assertEqual((14.0, "h"), self.initial_phase({"init_phase_length": "14", "init_phase_level": "h"}))


if __name__ == '__main__':
    unittest.main()
