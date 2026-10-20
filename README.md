# z2analyzer

How much of a ride can really stay in Zone 2?

On some slopes, even in the smallest gear and at the slowest cadence you can
actually hold, the power the road asks for is above your Zone 2 ceiling. Those
minutes are not a lack of discipline: the road made them. z2analyzer sorts
this out for you:

- **Before a ride**: give it a GPX and a time (or "cool" / "hot"). It tells you
  the share of the route you can ride in Zone 2, in kilometres and in time, and
  lists the passages where Zone 2 is impossible.
- **After a ride**: give it the FIT file from your bike computer. It splits the
  minutes you spent above Zone 2 between **the road** (Zone 2 was impossible
  there, even in the cool), **the heat** (possible in the cool, not that day)
  and **you** (it was possible).

Physics handles the slope. Your own rides handle the rest: TabPFN v2 learns,
from your FIT files, how your heart rate answers your power in the heat and
over the hours, the lowest cadence you really hold on a climb, and your speed
when you ride easy. Everything runs on your machine.

> Work in progress for the DEV Hacktoberfest Open-Source AI Challenge, week 3
> ("Slow and Steady"). Usage, results and limits are documented as they land.

## Status

- [x] Physics of the slope, gear recovered from speed and cadence, terrain tiles, profile, verdicts (tests)
- [ ] Learned limits with TabPFN v2 (heart rate, lowest cadence, easy speed)
- [ ] Command line and local page
- [ ] Validation on real rides

## License

Code: Apache License 2.0 (see `LICENSE` and `NOTICE`).

Built with PriorLabs-TabPFN. The TabPFN v2 weights are under the Prior Labs
License 1.1 (Apache 2.0 with an attribution clause, copy in `licenses/`).
