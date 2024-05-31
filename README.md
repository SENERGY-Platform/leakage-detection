# Leakage Detection

## Input 
| key                | type | description | 
|--------------------|------|-------------|   
| `Consumption`     | float | Numeric value that describes water consumption. |
| `Time`     | string | Timestamp of consumption value |



## Output 

| key | type | description | 
|--------------------|-------------|-----------------------------------------------------------| 
| `value`           | string | This string includes an information about execessive water consumption if this was detected during the last 5min. If nothing was detected this is just None. |


## Config options

No Config options!
