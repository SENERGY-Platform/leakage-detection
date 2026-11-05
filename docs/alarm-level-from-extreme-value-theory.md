# The alarm level comes from extreme value theory

## Scope

This applies to how a slot is judged in [main.py](../main.py) (`alarm_level` and `Operator.alarm_levels`), with slots of five minutes and 28 days of history. The figures below come from simulated households, not from real meters. They compare the variants with each other, and they do not predict the alarm levels of real meters. The continuous flow is a fixed rule and is not covered here; the README describes it.

## Decision

A slot is excessive above a level that comes from the tail of its hour: a generalized Pareto distribution fitted to the slots above the 90 % quantile, with a shape no lower than 0. The only parameter is `false_alarms_per_month`. This replaced DBSCAN with an epsilon from the knee of the nearest-neighbour distances and a 99.5 % quantile in November 2026.

## Why

- **The former judgement was a rank test.** The epsilon came from the distance to the nearest neighbour, but DBSCAN needs 10 slots within it, so every slot in the upper part of an hour counted as noise. What remained was the 99.5 % quantile: a slot was reported whenever it was among the two largest of its hour in 28 days, even one equal to the largest before. In the simulated households, about 2 such alarms came per hour of the day in 4 weeks, without any leak.
- **28 days cannot show events rare enough.** One false alarm a month means a probability of about 1 in 8,640 per slot, while an hour holds 336 slots. Only a model of the tail reaches that far beyond the slots seen.
- **Rejected: a fixed margin**, for example 1.5 times the largest slot and at least 10 litres. It would be the same for every household and hour.
- **Rejected: a shape estimated freely.** From the few dozen slots of a tail it often comes out below 0 by chance, which caps the level just above the largest slot seen. In the simulation that gave 5 to 14 false alarms in 4 weeks instead of about 1.
- **Rejected: one shape for all hours of a meter.** The hours differ in kind: flushes at night have a bounded tail, showers in the morning a long one. The shared estimate came out below 0, and so the lower bound set it to 0 anyway.
- **What it costs.** With the shape no lower than 0, the simulated households had 1 to 2 false alarms in 4 weeks at 1 a month, and 1 to 3 at 4 a month. In return the detection is less sensitive: in an hour with showers the level lay at about twice the largest shower.

## Before revisiting

Replay the history of several real meters and compare their false alarms with `false_alarms_per_month`, and check which bursts are found. If the detection turns out too insensitive, raise `false_alarms_per_month` before changing the method.
