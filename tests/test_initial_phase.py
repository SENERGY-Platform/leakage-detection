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

import tempfile
import unittest

from fake_timescale import Timescale
from main import CustomConfig, Operator


class TestInitialPhase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.sent = []

    def tearDown(self):
        self.dir.cleanup()

    def start(self):
        operator = Operator()
        operator.produce = self.sent.append
        operator.data_path = self.dir.name
        with Timescale().patched():
            operator.init(None, None, None, None, None, None,
                          config=CustomConfig({"init_phase_length": "2",
                                               "init_phase_level": "d"}))
        return operator

    def test_sends_nothing_before_the_first_value(self):
        self.start()
        # a message without a time of its own never reaches the export
        self.assertEqual([], self.sent)


if __name__ == '__main__':
    unittest.main()
