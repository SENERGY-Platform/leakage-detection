# Weekdays and weekends share one reference

## Scope

This applies to how a slot is judged in [main.py](../main.py): against all slots of the same hour of the day in the last 28 days, on any day of the week, with the 99.5 % quantile. If the history length or the quantile changes, the numbers below need to be checked again. The trade-off below comes from the code and was not measured on real meters.

The other operators handle the day of the week differently, and their reasons do not carry over here. [consumption_profile](https://github.com/SENERGY-Platform/consumption_profile) gives weekdays and weekends time windows of their own (`algo/dynamic_window_determination_user_profile.py`). [consumption_forecast_ml](https://github.com/SENERGY-Platform/consumption_forecast_ml) gives its model the day of the week as an input (`training.py`). In both, the day of the week sharpens a typical value. Here it would lower a bar.

## Decision

Slots are not split into weekdays and weekends. The split was considered in November 2026, and the reference was left as it is.

## Why

The judgement is a rank test. With linear interpolation, the 99.5 % quantile leaves only the largest slot above it in a list of up to 201 slots, and only the two largest in a list of up to 401. An hour of the day holds 28 × 12 = 336 slots. A slot is reported only if it is among the two largest of its hour and DBSCAN leaves it out. The shared reference therefore sets the bar at the highest normal consumption of that hour on any day.

A split would make the detection more sensitive in hours where the two types of day differ. On a weekday, a moderate consumption at a time when the household is usually away is currently hidden by higher weekend consumption in the same hour. Weekend mornings are hidden the same way by weekday mornings. A burst far above any normal consumption is reported either way.

Against the split:

- **More false alarms.** A weekday spent at home, for example in home office, sick, on leave or on a public holiday, would be judged against the quiet weekday reference.
- **A thin weekend reference.** 28 days hold 8 weekend days, so an hour has 96 weekend slots. Then the single largest of four weekends would be reported. Matching the 240 slots of the weekdays would need about 70 days of history for weekends. That history is older, and it is read from Timescale at every start.
- **The widget.** `last_consumptions` would have to show the slots of the same type of day, which is four weekend days in 14. That changes what the widget shows.

## Before revisiting

Replay the history of several real meters through both variants and compare the alarms. That shows whether the leaks found on weekdays outweigh the additional false alarms. If the split is adopted, it would follow consumption_profile: weekdays Monday to Friday and weekends, a longer history for weekends, and no public holidays.
