"""The pages draw no figure of the first and last 2 km of a route."""

from types import SimpleNamespace as NS

import numpy as np

from z2analyzer import page


def flat_plan(km_total: float):
    shares = NS(km_share=1.0, time_share=1.0, km_total=km_total, time_total_s=1800.0)
    return NS(shares=shares, shares_low=shares, shares_high=shares,
              stretches=[NS(end_m=km_total * 1000.0, possible=True)], passages=[])


def test_plan_page_figures_leave_out_the_ends():
    d = np.arange(0, 10_001, 10.0)
    # A hill in the first kilometre and nowhere else: it must not reach any figure.
    elev = np.where(d < 1000, 900.0, 100.0 + d / 1000)
    prof = NS(dist=d, elev=elev, lat=42.30 + d / 111_000, lon=np.full(d.size, 9.15))
    html = page.plan_page(flat_plan(10.0), "Test", "", prof)
    assert "900 m" not in html and "102 m" in html and "108 m" in html
    # The km axis is the route's own: the roadbook starts at km 2 and ends at km 8.
    assert '<span style="left:0%">2</span>' in html and "8.0 km" in html


def test_too_short_to_draw():
    d = np.arange(0, 3_001, 10.0)
    prof = NS(dist=d, elev=np.full(d.size, 50.0), lat=42.30 + d / 111_000, lon=np.full(d.size, 9.15))
    html = page.plan_page(flat_plan(3.0), "Short", "", prof)
    assert "Too short to draw" in html and "<polyline" not in html
