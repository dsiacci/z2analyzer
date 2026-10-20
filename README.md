# z2analyzer

How much of a ride can really stay in Zone 2?

On some slopes, even in your smallest gear and at the slowest cadence you can
actually hold, the power the road asks for is above your Zone 2 ceiling. Those
minutes are not a lack of discipline: the road made them. z2analyzer sorts
this out for you, before and after the ride.

- **Before**: give it a GPX and a start time (or a temperature). It tells you
  the share of the route you can ride in Zone 2, **in kilometres and in time**
  (climbs weigh more in time than in distance, and time is what a training plan
  counts), and lists the passages where Zone 2 is impossible.
- **After**: give it the FIT file from your bike computer. It splits the minutes
  you spent above Zone 2 between **the road** (impossible there, even in the
  cool), **the heat** (possible in the cool, not that day) and **you** (it was
  possible).

```text
$ z2analyzer review ride.fit --limits limits.json --z2-top 145
74 min above Zone 2: 31 imposed by the road, 9 by the heat, 34 mine.
Zone 2 was possible on 81 % of the kilometres, 64 % of the time.
```

*(the shape of the output, not a result: results are in [Results](#results))*

## How it works

**Physics for the slope.** The power needed to hold a speed on a grade:

    P = (m·g·v·(sin θ + Crr·cos θ) + ½·ρ·CdA·v³) / η

The slowest speed you can ride is your smallest gear (34×34 on a 50/34 with an
11-34 cassette: 2.1 m per pedal turn) times the lowest cadence you really hold.
Where the power at that speed is above your Zone 2 ceiling, Zone 2 is
impossible. The road's grade comes from a terrain model read from local files
(not from the noisy elevations of a GPX or a FIT), bridges and tunnels are
drawn straight, and the profile is cut into 100 m stretches.

**Your own rides for the rest.** Three things are personal and change with the
conditions, so they are learned from your FIT files with
[TabPFN](https://github.com/PriorLabs/TabPFN) v2, a tabular foundation model
that takes your rides as context and returns a distribution, not a single
number:

1. **Your heart rate** from your power (smoothed over one and five minutes:
   the heart lags the legs), the air temperature and the time already ridden.
   Inverted, it gives your Zone 2 ceiling in watts for any temperature and hour:
   heat and fatigue both lower it.
2. **The lowest cadence you really hold** in the smallest gear, from the grade
   and the time already spent climbing (a low quantile: the slowest you
   actually ride, not a theoretical minimum).
3. **Your speed riding easy**, from the grade and the power, to turn kilometres
   into time.

Each learned table keeps a low, middle and high value, so the verdicts come
with a range.

**A check that needs no model.** The gear you were in is recovered from speed
and cadence (speed ÷ (cadence × wheel) is the chainring/cog ratio). Where the
tool says Zone 2 was impossible, was the smallest gear already in? Where it
says Zone 2 was possible and your heart was above it, was there an easier gear
left?

## Usage

```bash
pip install "z2analyzer[model] @ git+https://github.com/dsiacci/z2analyzer"

# Once: terrain tiles and the bridges of your region (see Data below)
z2analyzer structures --bbox 41.3 8.5 43.1 9.6 --out bridges.json

# Your rides → one-second series without positions → your limits
z2analyzer prepare rides/*.fit --terrain dem/ --structures bridges.json --out prepared/
z2analyzer validate prepared/                     # how well the models predict rides they never saw
z2analyzer learn prepared/ --z2-top 145 --out limits.json

# Before a ride (temperature from the Météo-France forecast at the start time)
export Z2_MASS_KG=...                             # your mass: read from the environment, never stored or printed
z2analyzer plan route.gpx --limits limits.json --terrain dem/ --structures bridges.json --when "2026-10-24 08:30" --html plan.html

# After a ride
z2analyzer review ride.fit --limits limits.json --terrain dem/ --structures bridges.json --z2-top 145 --html ride.html
```

No power meter, or no rides yet? `z2analyzer fixed --ceiling-cool 150
--ceiling-hot 120 --cadence 50 --out limits.json` builds limits from three
numbers (a Zone 2 ceiling in the cool and at 30 °C, a lowest cadence).
Everything else is a parameter too: `--z2-top`, the gearing and the bike's mass
(`z2analyzer.physics.Bike`).

Outputs are shares, minutes, grades and multiples of your ceiling. Watts only
with `--watts`: a power, a speed and a grade on the same stretch are enough to
work out the rider's mass.

## What leaves your machine

- **Nothing from your rides.** FIT and GPX files are read locally; prepared
  rides keep no positions; terrain tiles are read from disk.
- **Weather**: one point rounded to 0.1° (about 11 km) and a date, sent to
  Open-Meteo. `--edge-temp` uses your computer's own sensor instead (warmed by
  the sun: up to 4 °C high on a hot ride).
- **Bridges and tunnels**: one request for a box you choose (a whole region),
  never for a track, sent to the OpenStreetMap Overpass API, once.
- **TabPFN**: the weights are downloaded once from Hugging Face; it runs on the
  CPU, with no account, no API and no cost per analysis.
- The local page (`--html`) loads nothing from anywhere: no fonts, no scripts,
  no map tiles (a tile server would learn which area you looked at). It draws
  the route's shape without its first and last 2 km.

## Data

| What | Source | Licence |
|---|---|---|
| Terrain, France | IGN, [RGE ALTI® 5 m](https://geoservices.ign.fr/rgealti), one archive per département, ESRI ASCII grids in Lambert-93 | Licence Ouverte 2.0 (Etalab) |
| Bridges and tunnels | © OpenStreetMap contributors, through the Overpass API | ODbL 1.0 |
| Air temperature | [Open-Meteo](https://open-meteo.com), Météo-France models (forecast, and the archive of past forecasts for past rides) | CC BY 4.0 |
| Model | TabPFN v2 regressor, [Prior-Labs/TabPFN-v2-reg](https://huggingface.co/Prior-Labs/TabPFN-v2-reg) | Prior Labs License 1.1 |
| FIT files | read with [fitdecode](https://github.com/polyvertex/fitdecode) 0.10.0 | MIT |

Outside France, any ESRI ASCII grid in Lambert-93 works today; other terrain
models (Copernicus DEM) need a small reader.

## Results

*(to come: the rides they were measured on, leave-one-ride-out errors, TabPFN
against the baselines, the gear check, and what failed)*

## Limits

- **Wind is not counted.** A headwind on a climb raises the power needed; a
  tailwind lowers it.
- The Zone 2 ceiling is the one of your own heart-rate zones (`--z2-top`),
  whatever they are based on.
- The numbers hold for one bike and one rider: the models learn from that
  rider's rides, and the gearing and masses are parameters.
- The heart rate also depends on sleep, fatigue and hydration, which no FIT
  file records: the ranges show part of that, not all of it.
- A finding about a route or a ride, not training advice and not medical
  advice.

## License

Code: Apache License 2.0 (see `LICENSE` and `NOTICE`).

**Built with PriorLabs-TabPFN.** The TabPFN v2 weights are under the Prior Labs
License 1.1 (Apache 2.0 with an attribution clause, copy in `licenses/`); they
are downloaded at run time and not included here.
