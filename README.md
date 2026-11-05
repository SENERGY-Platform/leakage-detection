# Leakage Detection

Reports a water consumption within five minutes that is far higher than in the same hour of the day on the days before, and water that has flowed for a day without the meter standing still.

## How it works

- **Slots:** The meter reading is interpolated linearly at every five-minute boundary between the readings around it. A slot gets the consumption between its two boundaries, so no consumption is lost and the result does not depend on the rate the meter reports at. A meter that reports seldom cannot show a short peak between two readings, though: it is spread over the slots in between.
- **Gaps and new meters:** Between two readings more than 2 hours apart, and where the reading falls, no slot is derived.
- **Hours of the day:** A slot belongs to the hour of the day in German time it starts in. Each hour keeps its slots of the last 28 days.
- **Judgement:** A slot is judged once the first reading after its end arrives, against the slots of its hour before it. It is excessive above their alarm level, which a normal slot exceeds about `false_alarms_per_month` times a month. The level comes from extreme value theory: the slots above the 90 % quantile of the hour are the tail, a generalized Pareto distribution is fitted to it with probability weighted moments, and the level is where that distribution reaches the probability of the false alarms. The shape of the tail is taken no lower than 0, an exponential tail, so the level lies above the largest slots seen rather than just above them. Slots are compared to 10 ml.
- **Quiet hours:** An hour with fewer than 10 slots in its tail is judged against the slots of the whole day. While the whole day has fewer than 10 in its tail, as in the first hours of a new meter, no slot is judged.
- **Continuous flow:** The meter stands still when its slots consume nothing for `rest_minutes`. If it has not stood still for `continuous_flow_hours`, the output reports it, once until it stands still again. A gap of more than 2 hours in the readings counts as standing still, so that an outage of the meter raises no alarm.
- **History:** At every start the operator reads the readings of the last 28 days from Timescale and derives the slots from them, so a restart or a new volume loses nothing. This needs a database connection (`TS_CONN` in the environment or `ts_conn` in the config); without one, or for an input whose history only Kafka has, it starts without history and logs a warning.
- **Initial phase:** Counts from the first value of the history, or from the first value received. A history as long as the initial phase ends it at once.

Design decisions and their reasons are in [docs/](docs/).

## Limitations

- **Sensitivity:** The alarm level lies well above the largest slots of an hour so far. In simulated households it lay at about twice the largest shower in hours with showers, so a leak that stays below that within five minutes is not reported as excessive. The method has not been checked on real meters yet. If it turns out too insensitive, raise `false_alarms_per_month` first. The figures and the reasons are in [docs/alarm-level-from-extreme-value-theory.md](docs/alarm-level-from-extreme-value-theory.md).
- **Meters that never stand still:** Where water is used around the clock, as in larger buildings, the meter may not stand still for `rest_minutes` within `continuous_flow_hours`, and the continuous flow is reported without a leak. There, set `continuous_flow_hours` to 0 or lower `rest_minutes`.
- **Restarts:** Whether the continuous flow has been reported is not kept. After a restart while the water still flows, it is reported once more.

## Input

| key | type | description |
|--------------------|------|-------------|
| `Consumption` | float | Meter reading in m³. |

The time of a reading is the time of its Kafka message.

## Output

One output per reading that completes at least one slot.

| key | type | description |
|--------------------|-------------|-----------------------------------------------------------|
| `value` | int | 1 if one of the slots the reading completes is excessive or the continuous flow is reported, 0 otherwise. |
| `timestamp` | string | UTC time of the reading, e.g. `2026-07-01T08:10:30.000Z`. |
| `message` | string | Only with `value` 1: a note about the excessive consumption, the continuous flow, or both. |
| `last_consumptions` | string | JSON array of `[end, litres, excessive]`, one per slot of the last 14 days in the hour of the day of the last completed slot. `end` is the end of the slot in German time as milliseconds since the epoch, that is, written as if German time were UTC; `excessive` is 1 if the slot lies above the current alarm level of the hour, 0 otherwise. |
| `time_window` | string | That hour of the day in German time, e.g. `08:00:00-09:00:00`. |
| `initial_phase` | string | During the initial phase a note until when it lasts, in the output that ends it an empty string, absent afterwards. |

During the initial phase, and in the output that ends it, `value` is 0 and `message`, `last_consumptions` and `time_window` are empty.

## Config options

| key | type | description |
|--------------------|-------------|-----------------------------------------------------------|
| `logger_level` | str | default: "warning" |
| `init_phase_length` | float | Length of the initial phase in units of `init_phase_level`. default: 2 |
| `init_phase_level` | string | Unit of `init_phase_length` as a pandas Timedelta unit, e.g. "d" or "h". default: "d" |
| `false_alarms_per_month` | float | How often a month a normal slot may be reported as excessive; above 0. The higher, the lower the alarm level. default: 1 |
| `continuous_flow_hours` | float | How long the water may flow without the meter standing still before it is reported; 0 turns this off. default: 24 |
| `rest_minutes` | float | How long the meter has to consume nothing to stand still; above 0. default: 60 |

Example: If init_phase_length is 14 and init_phase_level is "d" then the initial phase lasts 14 days.

The operator keeps the start of its initial phase in `/opt/data`, which should be a volume. `operator-manifest.json` declares the inputs, outputs and config options; the CI regenerates it from `main.py` with `python -m operator_lib manifest main.py` before each release.
