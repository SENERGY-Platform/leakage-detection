# Leakage Detection

Reports a water consumption within five minutes that is far higher than in the same hour of the day on the days before.

## How it works

- **Slots:** The meter reading is interpolated linearly at every five-minute boundary between the readings around it. A slot gets the consumption between its two boundaries, so no consumption is lost and the result does not depend on the rate the meter reports at. A meter that reports seldom cannot show a short peak between two readings, though: it is spread over the slots in between.
- **Gaps and new meters:** Between two readings more than 2 hours apart, and where the reading falls, no slot is derived.
- **Hours of the day:** A slot belongs to the hour of the day in German time it starts in. Each hour keeps its slots of the last 28 days.
- **Judgement:** A slot is judged once the first reading after its end arrives, from 10 slots of its hour on. It is excessive if a DBSCAN clustering of the slots of its hour leaves it out and it lies above their 99.5 % quantile.
- **History:** At every start the operator reads the readings of the last 28 days from Timescale and derives the slots from them, so a restart or a new volume loses nothing. This needs a database connection (`TS_CONN` in the environment or `ts_conn` in the config); without one, or for an input whose history only Kafka has, it starts without history and logs a warning.
- **Initial phase:** Counts from the first value of the history, or from the first value received. A history as long as the initial phase ends it at once.

Design decisions and their reasons are in [docs/](docs/).

## Input

| key | type | description |
|--------------------|------|-------------|
| `Consumption` | float | Meter reading in m³. |

The time of a reading is the time of its Kafka message.

## Output

One output per reading that completes at least one slot.

| key | type | description |
|--------------------|-------------|-----------------------------------------------------------|
| `value` | int | 1 if one of the slots the reading completes is excessive, 0 otherwise. |
| `timestamp` | string | UTC time of the reading, e.g. `2026-07-01T08:10:30.000Z`. |
| `message` | string | Only with `value` 1: a note about the excessive consumption. |
| `last_consumptions` | string | JSON array of `[end, litres, excessive]`, one per slot of the last 14 days in the hour of the day of the last completed slot. `end` is the end of the slot in German time as milliseconds since the epoch, that is, written as if German time were UTC; `excessive` is 1 or 0. |
| `time_window` | string | That hour of the day in German time, e.g. `08:00:00-09:00:00`. |
| `initial_phase` | string | During the initial phase a note until when it lasts, in the output that ends it an empty string, absent afterwards. |

During the initial phase, and in the output that ends it, `value` is 0 and `message`, `last_consumptions` and `time_window` are empty.

## Config options

| key | type | description |
|--------------------|-------------|-----------------------------------------------------------|
| `logger_level` | str | default: "warning" |
| `init_phase_length` | float | Length of the initial phase in units of `init_phase_level`. default: 2 |
| `init_phase_level` | string | Unit of `init_phase_length` as a pandas Timedelta unit, e.g. "d" or "h". default: "d" |

Example: If init_phase_length is 14 and init_phase_level is "d" then the initial phase lasts 14 days.

The operator keeps the start of its initial phase in `/opt/data`, which should be a volume. `operator-manifest.json` declares the inputs, outputs and config options; the CI regenerates it from `main.py` with `python -m operator_lib manifest main.py` before each release.
