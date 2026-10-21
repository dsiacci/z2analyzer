"""The web app end to end on made-up data: a new analysis of each kind, the list, the totals, the refusals."""

import http.client
import os
import re
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np

from fitwriter import write_fit
from test_cli import gpx, ramp_ride, terrain
from z2analyzer import cli
from z2analyzer.physics import Rider
from z2analyzer.dem import Terrain
from z2analyzer.web import App, Settings, Store, home_page, make_server, parse_multipart, run, watch_folder


def form(fields: dict, file: tuple[str, bytes] | None = None) -> tuple[str, bytes]:
    b = "----z2test" + secrets.token_hex(8)
    out = b"".join(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items())
    if file:
        out += (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{file[0]}"\r\n'
                f"Content-Type: application/octet-stream\r\n\r\n").encode() + file[1] + b"\r\n"
    return f"multipart/form-data; boundary={b}", out + f"--{b}--\r\n".encode()


def test_parse_multipart_keeps_binary_files():
    data = bytes(np.random.default_rng(1).integers(0, 256, 5000, dtype=np.uint8)) + b"\r\n--\r\n\r\n\x00"
    ctype, body = form({"kind": "review", "token": "t"}, ("2026-10-04.fit", data))
    fields, files = parse_multipart(ctype, body)
    assert fields == {"kind": "review", "token": "t"}
    assert files["file"] == ("2026-10-04.fit", data)


def test_store_keeps_the_newest_first(tmp_path):
    st = Store(tmp_path)
    a = st.create("plan", "../../etc/passwd", b"<gpx/>", {})
    b = st.create("review", "ride one.FIT", b"x", {})
    assert [m["id"] for m in st.all()] == [b["id"], a["id"]]
    assert a["file"] == "passwd" and b["file"] == "ride-one.FIT"
    assert st.get("../x") is None
    st.delete(a["id"])
    assert [m["id"] for m in st.all()] == [b["id"]]


class Client:
    def __init__(self, port):
        self.port = port

    def request(self, method, path, body=None, ctype=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        c.request(method, path, body=body, headers={"Content-Type": ctype} if ctype else {})
        r = c.getresponse()
        out = (r.status, r.getheader("Location"), r.read().decode("utf-8"))
        c.close()
        return out

    def until_done(self, path):
        for _ in range(200):
            code, _, page = self.request("GET", path)
            if "Analysing…" not in page:
                return code, page
            time.sleep(0.1)
        raise AssertionError(f"{path} never finished")


def test_web_app_end_to_end(tmp_path):
    dem = terrain(tmp_path)
    route, lat = gpx(tmp_path)
    lim = tmp_path / "limits.json"
    assert cli.main(["fixed", "--ceiling-cool", "150", "--ceiling-hot", "120", "--cadence", "50", "--mass-kg", "75",
                     "--out", str(lim)]) == 0
    settings = Settings(limits=str(lim), terrain=str(dem), structures=None, z2_top=145.0, edge_temp=True)
    app = App(Store(tmp_path / "data"), settings, Rider(mass_kg=75.0))
    httpd = make_server(app, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    web = Client(httpd.server_address[1])
    try:
        code, _, home = web.request("GET", "/")
        assert code == 200 and "New analysis" in home and "None yet" in home
        code, _, new = web.request("GET", "/new")
        token = re.search(r'name="token" value="([^"]+)"', new).group(1)
        assert "Before a ride" in new and "After a ride" in new

        # Before a ride: a GPX, in the cool of the rides.
        ctype, body = form({"token": token, "kind": "plan", "name": "Test climb", "conditions": "cool"},
                           ("route.gpx", route.read_bytes()))
        code, where, _ = web.request("POST", "/new", body, ctype)
        assert code == 303 and where.startswith("/a/")
        code, page = web.until_done(where)
        assert code == 200 and "of the kilometres" in page and "Your analyses" in page and "judged with your" in page
        assert "https://" not in page and "http://" not in page.replace("http://www.w3.org", "")

        # After a ride: a FIT up the same road, then a second one; the home page adds them up.
        rides = []
        for day in (27, 28):
            fit = tmp_path / f"2026-09-{day}.fit"
            write_fit(fit, datetime(2026, 9, day, 6, 0, tzinfo=timezone.utc), ramp_ride(lat))
            ctype, body = form({"token": token, "kind": "review"}, (fit.name, fit.read_bytes()))
            code, where, _ = web.request("POST", "/new", body, ctype)
            assert code == 303
            code, page = web.until_done(where)
            assert "Who made the" in page and "the road" in page
            rides.append(where)
            if day == 27:
                assert "Your rides so far" not in web.request("GET", "/")[2]
        meta = app.store.get(rides[0].split("/")[-1])
        assert meta["summary"]["road"] > 0.8 * meta["summary"]["above"] > 0
        home = web.request("GET", "/")[2]
        assert "Test climb" in home and "min above Zone 2" in home and "Your rides so far" in home and "on 2 rides" in home
        assert home.index("Sun 27 Sep") > home.index("Mon 28 Sep")  # the newest first

        # What the app refuses, and says so.
        ctype, body = form({"token": token, "kind": "review"}, ("route.gpx", route.read_bytes()))
        code, _, page = web.request("POST", "/new", body, ctype)
        assert code == 400 and "is not a FIT file" in page
        ctype, body = form({"kind": "review"}, ("a.fit", b"x" * 20))
        assert web.request("POST", "/new", body, ctype)[0] == 403
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        ctype, body = form({"token": token, "kind": "plan", "conditions": "forecast", "day": yesterday, "time": "08:30"},
                           ("route.gpx", route.read_bytes()))
        code, _, page = web.request("POST", "/new", body, ctype)
        assert code == 400 and "in the past" in page
        indoor = tmp_path / "indoor.fit"
        write_fit(indoor, datetime(2026, 9, 29, 6, 0, tzinfo=timezone.utc), ramp_ride(lat)[:200], manufacturer=260)
        ctype, body = form({"token": token, "kind": "review"}, (indoor.name, indoor.read_bytes()))
        code, where, _ = web.request("POST", "/new", body, ctype)
        code, page = web.until_done(where)
        assert "Not analysed" in page and "indoor ride" in page

        # Deleting an analysis.
        code, where, _ = web.request("POST", rides[0] + "/delete", f"token={token}", "application/x-www-form-urlencoded")
        assert code == 303 and where == "/"
        assert web.request("GET", rides[0])[0] == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.stop()


def test_new_rides_in_the_watched_folder_are_judged_on_their_own(tmp_path):
    dem = terrain(tmp_path)
    _, lat = gpx(tmp_path)
    lim = tmp_path / "limits.json"
    cli.main(["fixed", "--ceiling-cool", "150", "--ceiling-hot", "120", "--cadence", "50", "--mass-kg", "75", "--out", str(lim)])
    rides = tmp_path / "rides"
    rides.mkdir()
    write_fit(rides / "monday.fit", datetime(2026, 9, 27, 6, 0, tzinfo=timezone.utc), ramp_ride(lat))
    write_fit(rides / "indoor.fit", datetime(2026, 9, 28, 6, 0, tzinfo=timezone.utc), ramp_ride(lat)[:200], manufacturer=260)
    (rides / "notes.txt").write_text("not a ride")
    write_fit(rides / "copying.fit", datetime(2026, 9, 29, 6, 0, tzinfo=timezone.utc), ramp_ride(lat))
    old = time.time() - 120
    for f in ("monday.fit", "indoor.fit", "notes.txt"):
        os.utime(rides / f, (old, old))
    store = Store(tmp_path / "data")
    seen = tmp_path / "data" / "watched.json"
    new = watch_folder(store, rides, seen)
    assert [m["file"] for m in new] == ["monday.fit"]  # not the indoor ride, nor the file still being copied
    os.utime(rides / "copying.fit", (old, old))
    assert [m["file"] for m in watch_folder(store, rides, seen)] == ["copying.fit"]
    assert watch_folder(store, rides, seen) == []

    settings = Settings(limits=str(lim), terrain=str(dem), structures=None, z2_top=145.0, edge_temp=True, watch=str(rides))
    m = run(store, new[0]["id"], settings, Rider(mass_kg=75.0), Terrain(dem), None)
    assert m["status"] == "done" and m["summary"]["road"] > 0
    page = home_page(store, settings)
    assert "From your rides folder, judged on its own" in page and "“rides”" in page
