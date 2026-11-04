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

from unittest import mock

import pandas as pd


class Timescale:
    # The history of the input that the operator reads at its start, as the library returns it:
    # one frame with the UTC time without a timezone and the meter reading.

    def __init__(self):
        self.rows = []

    def write(self, at, consumption):
        self.rows.append((pd.Timestamp(at).tz_convert("UTC").tz_localize(None), consumption))

    def patched(self):
        # the reader of the operator for an init within the with block
        frame = pd.DataFrame(self.rows, columns=["time", "Consumption"])
        return mock.patch("main.provide_historic_frames_local", return_value=[frame])
