# Leakage Detection

## Input 
| key                | type | description | 
|--------------------|------|-------------|   
| `Consumption`     | float | Numeric value that describes water consumption. |
| `Time`     | string | Timestamp of consumption value |



## Output 

| key | type | description | 
|--------------------|-------------|-----------------------------------------------------------| 
| `value`           | int | 0 if excessive consumption was detected during the last 5min; 1 otherwise. |
| `timestamp`           | string | This string includes the timestamp of the last datapoint. |
| `message`           | string | This string includes an information about execessive water consumption if this was detected during the last 5min. If nothing was detected this is just None. |
| `last_consumptions`           | string | This string represents a pandas dataframe in which all 5min-consumptions from the last days during the current hour of the day are stored. |
| `time_window`           | string | This string includes the time boundaries from the current hour of the day. |
| `initial_phase`           | string | This string includes an information about whether the operator is in an initial learning phase or not. |


## Config options

| key | type | description | 
|--------------------|-------------|-----------------------------------------------------------| 
| `logger_level`           | str | default: "warning" |
| `init_phase_length`           | int |  |
| `init_phase_level`           | string | |

Example: If init_phase_length is 14 and init_phase_level is "d" then the initial phase lasts 14 days.
