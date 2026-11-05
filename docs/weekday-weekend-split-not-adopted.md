# Weekdays and weekends share one reference

## Scope

This applies to how a slot is judged in [main.py](../main.py): against the alarm level of its hour, which comes from the slots of the same hour of the day in the last 28 days, on any day of the week. The alarm level itself is explained in [alarm-level-from-extreme-value-theory.md](alarm-level-from-extreme-value-theory.md). If the history length or the alarm level changes, the reasoning below needs to be checked again. The trade-off below comes from the code and was not measured on real meters.

The other operators handle the day of the week differently, and their reasons do not carry over here. [consumption_profile](https://github.com/SENERGY-Platform/consumption_profile) gives weekdays and weekends time windows of their own (`algo/dynamic_window_determination_user_profile.py`). [consumption_forecast_ml](https://github.com/SENERGY-Platform/consumption_forecast_ml) gives its model the day of the week as an input (`training.py`). In both, the day of the week sharpens a typical value. Here it would lower a bar.

## Decision

Slots are not split into weekdays and weekends. The split was considered in November 2026, and the reference was left as it is. The decision was kept when the alarm level was moved to extreme value theory in the same month.

## Why

The alarm level of an hour comes from the tail of its slots, the largest tenth of them. In an hour that all days share, the type of day that consumes more in that hour fills the tail. The level therefore lies above the highest normal consumption of that hour on any day.

A split would make the detection more sensitive in hours where the two types of day differ. On a weekday, a moderate consumption at a time when the household is usually away stays below a level that weekend consumption in the same hour has set. Weekend mornings are hidden the same way by weekday mornings. A burst far above any normal consumption is reported either way.

Against the split:

- **More false alarms.** A weekday spent at home, for example in home office, sick, on leave or on a public holiday, would be judged against the quiet weekday reference.
- **A thin weekend reference.** 28 days hold 8 weekend days, so an hour has 96 weekend slots, and about 10 of them lie in its tail. That is the limit below which an hour is judged against the whole day. Weekend hours would often fall back to the whole day, or they would need about 70 days of history. That history is older, and it is read from Timescale at every start.
- **The widget.** `last_consumptions` would have to show the slots of the same type of day, which is four weekend days in 14. That changes what the widget shows.

## Before revisiting

Replay the history of several real meters through both variants and compare the alarms. That shows whether the leaks found on weekdays outweigh the additional false alarms. If the split is adopted, it would follow consumption_profile: weekdays Monday to Friday and weekends, a longer history for weekends, and no public holidays.
