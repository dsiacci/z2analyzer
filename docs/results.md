# Results on one rider's rides

Ten outdoor rides with a power meter, from 14 August to 4 October 2026, in
Corsica: 24.7 hours, from 19 to 109 km, ridden at dawn (air between 19 and
29 °C, median of each ride). One rider, one road bike (50/34, 11-34), a top of
Zone 2 at 145 bpm. Every number below comes from `z2analyzer validate`,
`tools/experiments.py` and `tools/proof.py`; none of them is in watts, on
purpose (with a speed and a grade, watts give away the rider's mass).

## 1. The data

- **Terrain against the barometer.** The total climbing of the rebuilt profile
  (IGN RGE ALTI 5 m, bridges and tunnels drawn straight) is within 2 to 9 % of
  the bike computer's barometric altitude on every ride.
- **Air against the computer's sensor.** The Météo-France temperature (one
  point rounded to 0.1°, moved to the road's elevation) and the computer's own
  sensor agree within a degree at dawn; on the hottest ride the sensor read
  4 °C high in the sun. The models use the air temperature, so that a forecast
  can stand in for it before a ride.
- **Gears.** Speed divided by cadence falls on the cassette's ratios (34×30 and
  34×27 show as clear peaks), and the smallest gear is engaged from almost
  never (0.3 % of the pedalling time) to almost two thirds of it, depending on
  the ride.

## 2. Which model for which table

Each ride predicted by models trained on the nine others (30-second windows).
The rule, fixed before looking: the lowest pinball loss on the quantile the
verdicts use (the median heart rate, the 10 % cadence); mean errors are shown
because they are easier to read.

| Table | TabPFN v2 | Linear regression | Constant |
|---|---|---|---|
| Heart rate, 2,894 windows: mean error / 10–90 % interval holds | 8.1 bpm / 59 % | **6.5 bpm** / 74 % | 16.2 bpm / 80 % |
| Lowest cadence in 34×34, 360 windows on climbs: pinball loss at 10 % | 1.82 | **1.59** | 1.60 |
| Speed, 2,894 windows: mean error | **0.70 m/s** | 1.20 m/s | 2.29 m/s |

**TabPFN wins the speed** by a wide margin: speed against grade and power is
curved (air, descents where the rider brakes). **It loses the heart rate and
the cadence.** The lowest cadence is almost a constant (a line barely beats
one). The heart rate is where the most interesting failure was:

- At the same power, the heart's level moves from one day to the next: from
  13 bpm below to 7 bpm above its usual level, measured on the first half of
  each ride (−17 to +8 over whole rides, in the table below). Sleep, fatigue
  and water are not in a FIT file.
- Each ride had its own range of temperatures, so TabPFN could all but
  recognise a ride by its temperature and copy that day's level. A line
  cannot. Its 10–90 % interval held the truth only 59 % of the time.
- Variants (60-second windows, `tools/experiments.py`), the day's level
  measured on the first half of each ride and the second half predicted:

| Heart rate, second half of each ride | Mean error | Pinball at 50 % |
|---|---|---|
| Linear regression | **4.68 bpm** | **2.34** |
| Linear regression, training days brought to a common level | 4.75 bpm | 2.37 |
| TabPFN, training days brought to a common level | 4.85 bpm | 2.43 |
| Linear + TabPFN on its residuals, common level | 4.88 bpm | 2.44 |
| TabPFN, raw | 6.88 bpm | 3.44 |

  Once the days' levels are removed, TabPFN catches up with the line, but
  does not pass it. Across days (before a ride, nothing measured yet): line
  6.2 bpm, line + TabPFN on residuals 7.3, TabPFN 8.5.

**Chosen:** a linear regression for the heart rate and the lowest cadence,
TabPFN for the speed (`learn --models hr=linear cadence=linear speed=tabpfn`).
On another rider's rides the scores may come out differently: `validate`
shows them.

Two consequences in the tool:

- After a ride, the day's level is measured on the ride's own steady stretches
  and the Zone 2 ceiling moves with it.
- Before a ride, the ceiling's range is the spread of that level between rides
  left out (10 % to 90 %: −6 to +7 bpm), not the model's own interval.

One more failure, fixed: the line for the lowest cadence kept falling with the
grade, to 40 rpm on the steepest slopes, while the steepest climbs show nothing
much below 49. Every learned table now stops at the 1st and 99th percentiles of
what the rides cover (grades of 3 to 15 %, air from 18.5 to 32.8 °C, up to
5 h 25 ridden) and holds its edge value beyond.

## 3. Every ride judged by limits that never saw it

`tools/proof.py`: for each ride, limits learned from the nine others, then the
verdict on that ride. Zone 2 up to 145 bpm; the road when even the slowest
pace was above the ceiling in the cool (20 °C, the coolest tenth of these
rides); the heat when it was possible in the cool but not that day; me when it
was possible that day.

| Date | Moving | Air (median) | Heart vs usual | Above Zone 2 | The road | The heat | Me | Zone 2 possible: km / time |
|---|---|---|---|---|---|---|---|---|
| 14/08 | 88 min | 22 °C | −8 bpm | 27 min | 0 | 0 | 27 | 100 % / 99 % |
| 16/08 | 110 min | 24 °C | −17 bpm | 4 min | 0 | 0 | 4 | 100 % / 100 % |
| 19/08 | 340 min | 29 °C | +8 bpm | 201 min | 27 | 8 | 166 | 92 % / 85 % |
| 24/08 | 63 min | 25 °C | −6 bpm | 18 min | 0 | 0 | 18 | 99 % / 98 % |
| 29/08 | 89 min | 25 °C | +4 bpm | 47 min | 25 | 3 | 19 | 85 % / 66 % |
| 31/08 | 62 min | 23 °C | −7 bpm | 16 min | 0 | 0 | 16 | 100 % / 99 % |
| 13/09 | 188 min | 21 °C | −6 bpm | 34 min | 0 | 0 | 34 | 100 % / 99 % |
| 20/09 | 180 min | 22 °C | +2 bpm | 84 min | 1 | 1 | 82 | 98 % / 97 % |
| 27/09 | 169 min | 19 °C | +2 bpm | 86 min | 1 | 0 | 85 | 99 % / 98 % |
| 04/10 | 191 min | 24 °C | +1 bpm | 119 min | 45 | 5 | 70 | 88 % / 72 % |
| **10 rides** | **1,480 min** | | | **637 min** | **98** | **17** | **522** | |

**The road made 15 % of the minutes above Zone 2, the heat 3 %, the rider
82 %.** The road matters on the steep rides (29 August, 4 October: up to 11 %
for 900 m in the smallest gear); on the 58 km loop ridden on 13, 20 and
27 September, Zone 2 was possible on 98 to 100 % of the kilometres.

With one fixed ceiling and one fixed cadence instead (no heat, no hours ridden,
no day level): the road 64 min, the heat 0, me 572.

Read with care:

- The tool counts minutes, not intensity. On a stretch where the road made
  Zone 2 impossible, it does not say how far above it the rider had to go.
- "Me" can mean pedalling slower in the same gear. On 14 August, three
  quarters of the seconds above Zone 2 were already in 34×34, but at 54 rpm on
  grades where this rider holds 46 rpm when he has to.
- The heat share is small for two reasons: these rides start at dawn, and a
  hot day's extra heart rate beyond what the model learned about temperature
  lands in that day's level, which moves the cool ceiling too.

## 4. The check that needs no model

The gear, recovered from speed and cadence, over the ten rides:

- Where Zone 2 was possible and the heart was above it, an easier gear was
  still there **91 %** of the time (28,248 seconds).
- Where Zone 2 was impossible, the smallest gear was engaged only 43 % of the
  time (8,890 seconds): those climbs were often ridden harder than the
  slowest pace, which the tool does not ask about.

## 5. Before a ride

The 4 October loop (59.6 km, 1,085 m of climbing on the terrain model),
planned for Saturday 24 October at 08:30 with the Météo-France forecast (15 °C
at the start, counted as 19 °C, the coolest these rides cover): Zone 2 possible
on **89 % of the kilometres but 70 % of the time**; 11 passages, 6.8 km, 1 h 02
at the slowest pace; the longest, 1.4 km at 10.9 %, asks 1.7 times the ceiling.
At 30 °C: 86 % and 65 %. The 58 km loop the same morning: 100 % and 99 %.
