# z2analyzer

How much of a ride can really stay in Zone 2?

On some slopes, even in your smallest gear and at the slowest cadence you can
actually hold, the power the road asks for is above your Zone 2 ceiling. Those
minutes are the road's, not yours. z2analyzer sorts out which minutes are
which, before and after the ride, so you don't have to read the curves.

- **Before**: give it a GPX and a start time (or a temperature). It tells you
  the share of the route you can ride in Zone 2, **in kilometres and in time**
  (climbs weigh more in time than in distance, and time is what a training plan
  counts), and lists the passages where Zone 2 is impossible.
- **After**: give it the FIT file from your bike computer. It splits the minutes
  you spent above Zone 2 between **the road** (impossible there, even in the
  cool), **the heat** (possible in the cool, not that day) and **you** (it was
  possible).
- **Overnight**: point `z2analyzer watch` at the folder where your FIT files
  land. Every new ride gets its verdict, and the two lines wait for you in
  `morning.md`.

```text
$ z2analyzer review ride.fit --limits limits.json --z2-top 145
119 min above Zone 2: 45 imposed by the road, 5 by the heat, 70 mine.
Zone 2 was possible on 88 % of the kilometres, 72 % of the time.

$ z2analyzer plan loop.gpx --limits limits.json --when "2026-10-24 08:30"
Zone 2 is possible on 89 % of the kilometres, but 70 % of the time.
11 passages where it is not: 6.8 km, 1 h 02 even at your slowest pace.
```

*(the same 59.6 km loop: a ride on 4 October, then planned for a Saturday
morning; more in [Results](#results))*

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
with a range. `validate` scores TabPFN, a linear regression and a constant on
rides they never saw, and `learn --models` keeps the winner of each table: on
the rides it was built on, TabPFN for the speed and a line for the rest (see
[Results](#results)).

**The day's level.** At the same power, the heart does not sit at the same
level every day. After a ride, z2analyzer measures that day's level on the
ride's own steady stretches and moves the ceiling with it; before a ride, the
range it gives is how far that level moved between your rides.

**A check that needs no model.** The gear you were in is recovered from speed
and cadence (speed ÷ (cadence × wheel) is the chainring/cog ratio). Where the
tool says Zone 2 was impossible, was the smallest gear already in? Where it
says Zone 2 was possible and your heart was above it, was there an easier gear
left?

## Usage

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu  # optional: the CPU build, a much smaller download
pip install "z2analyzer[model] @ git+https://github.com/dsiacci/z2analyzer"

# Once: terrain tiles and the bridges of your region (see Data below)
z2analyzer structures --bbox 41.3 8.5 43.1 9.6 --out bridges.json

# Your rides → one-second series without positions → your limits
z2analyzer prepare rides/*.fit --terrain dem/ --structures bridges.json --out prepared/
z2analyzer validate prepared/ --out validate.json # how well each model predicts rides it never saw
z2analyzer learn prepared/ --z2-top 145 --models hr=linear cadence=linear speed=tabpfn \
    --validation validate.json --out limits.json  # the winner of each table on your rides

# Before a ride (temperature from the Météo-France forecast at the start time)
export Z2_MASS_KG=...                             # your mass: read from the environment, never stored or printed
z2analyzer plan route.gpx --limits limits.json --terrain dem/ --structures bridges.json --when "2026-10-24 08:30" --html plan.html

# After a ride
z2analyzer review ride.fit --limits limits.json --terrain dem/ --structures bridges.json --z2-top 145 --html ride.html

# Or let it do that by itself: every new FIT in the folder, judged once (run it from cron, or add --every 30)
z2analyzer watch ~/rides --limits limits.json --terrain dem/ --structures bridges.json --z2-top 145
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

- **No ride file and no track.** FIT and GPX files are read locally; prepared
  rides keep no positions; terrain tiles are read from disk.
- **Weather**: one point rounded to 0.1° (about 11 km) and a date, sent to
  Open-Meteo. `--edge-temp` uses your bike computer's sensor instead (warmed
  by the sun: up to 4 °C high on a hot ride).
- **Bridges and tunnels**: one request for a box you choose (a whole region),
  never for a track, sent to the OpenStreetMap Overpass API, once.
- **TabPFN**: the weights are downloaded once from Hugging Face; it runs on the
  CPU, with no account, no API and no cost per analysis.
- The local page (`--html`) loads nothing from anywhere: its fonts travel
  inside it, there are no scripts and no map tiles (a tile server would learn
  which area you looked at). Its figures (the route's shape, the profile, the
  climbs) leave out the first and last 2 km, so a screenshot doesn't show
  where you start; the numbers still count the whole route.

## Data

| What | Source | Licence |
|---|---|---|
| Terrain, France | IGN, [RGE ALTI® 5 m](https://geoservices.ign.fr/rgealti), one archive per département, ESRI ASCII grids in Lambert-93 | Licence Ouverte 2.0 (Etalab) |
| Bridges and tunnels | © OpenStreetMap contributors, through the Overpass API | ODbL 1.0 |
| Air temperature | [Open-Meteo](https://open-meteo.com), Météo-France models (forecast, and the archive of past forecasts for past rides) | CC BY 4.0 |
| Model | TabPFN v2 regressor, [Prior-Labs/TabPFN-v2-reg](https://huggingface.co/Prior-Labs/TabPFN-v2-reg) | Prior Labs License 1.1 |
| FIT files | read with [fitdecode](https://github.com/polyvertex/fitdecode) 0.10.0 | MIT |
| Fonts of the local pages | Saira Extra Condensed and Saira Semi Condensed, embedded in each page | SIL Open Font License 1.1 |

Outside France, any ESRI ASCII grid in Lambert-93 works today; other terrain
models (Copernicus DEM) need a small reader.

## Results

On one rider's ten outdoor rides with a power meter (Corsica, August to
October 2026, 24.7 hours), every ride judged with limits learned from the
nine others: **637 minutes above Zone 2, of which the road made 98, the heat
17 and the rider 522.** Where Zone 2 was possible and the heart was above it,
an easier gear was still there 91 % of the time.

TabPFN won one of its three jobs: it predicts the speed with 40 % less error
than a linear regression (0.70 against 1.20 m/s) on rides it never saw. A linear
regression won the heart rate (6.5 against 8.1 bpm) and, barely, the lowest
cadence; the reason, and what it changed in the tool, is in
[docs/results.md](docs/results.md), with every table and what failed.

## Limits

- **Wind is not counted.** A headwind on a climb raises the power needed; a
  tailwind lowers it.
- The Zone 2 ceiling is the one of your own heart-rate zones (`--z2-top`),
  whatever they are based on.
- The numbers hold for one bike and one rider: the models learn from that
  rider's rides, and the gearing and masses are parameters.
- The heart rate also depends on sleep, fatigue and hydration, which no FIT
  file records: after a ride the day's level absorbs it, before a ride the
  range shows it.
- The models only know the conditions of your rides. Outside them (a colder
  morning, a steeper grade, a longer ride) the tables hold their edge value
  instead of extrapolating, the lowest cadence never drops below the slowest
  your rides show, and `plan` says when the forecast is colder than any ride
  it learned from.
- The heat share only counts what the model learned about temperature across
  your rides; the rest of a hot day's extra heart rate lands in that day's
  level.
- It counts minutes, not intensity: where the road made Zone 2 impossible, it
  does not say how far above it you had to go.
- A finding about a route or a ride, not training advice and not medical
  advice.

## License

Code: Apache License 2.0 (see `LICENSE` and `NOTICE`).

**Built with PriorLabs-TabPFN.** The TabPFN v2 weights are under the Prior Labs
License 1.1 (Apache 2.0 with an attribution clause, copy in `licenses/`); they
are downloaded at run time and not included here.
