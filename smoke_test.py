"""Headless smoke test: executes app.py via Streamlit's AppTest runner.

Usage: python smoke_test.py
Asserts the dashboard renders all 5 boxes, and that changing the station /
sliders reruns cleanly without exceptions.
"""
from __future__ import annotations

import sys

# Console may be cp1252 (Windows) — allow emoji in printed output.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from streamlit.testing.v1 import AppTest

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {msg}")
    if not cond:
        FAILURES.append(msg)


at = AppTest.from_file("app.py", default_timeout=180)
at.run()

check(not at.exception, f"initial run has no exception ({at.exception})")

subheaders = [s.value for s in at.subheader]
print("subheaders:", subheaders)
for box, key in [("Box 1", "Box 1"), ("Box 2", "Box 2"), ("Box 3", "Box 3"),
                 ("Box 4", "Box 4"), ("Box 5", "Box 5")]:
    check(any(key in s for s in subheaders), f"{box} rendered")

# Station search interaction --------------------------------------------
station_sb = at.selectbox(key="station")
stations = list(station_sb.options)
check(len(stations) > 1, f"station search has options ({len(stations)})")
target = stations[-1]
station_sb.select(target)
at.run()
check(not at.exception, f"after selecting station '{target}' ({at.exception})")
check(any(target in s.value for s in at.subheader),
      "Box 1 header shows newly selected station")

# Slider override ---------------------------------------------------------
wl_sliders = [s for s in at.slider if s.label.startswith("River Water Level")]
check(len(wl_sliders) == 1, "water level slider present")
if wl_sliders:
    s = wl_sliders[0]
    bumped = round(float(s.max), 2)
    s.set_value(bumped)
    at.run()
    check(not at.exception, f"after slider override to {bumped} ({at.exception})")

# Search log box ----------------------------------------------------------
check(len(at.text_input) >= 1, "log search input present")
if at.text_input:
    at.text_input(key="log_search").set_value("Danger")
    at.run()
    check(not at.exception, "after log search 'Danger'")

# Date range filter -------------------------------------------------------
di = at.date_input(key="f_dates")
check(di is not None, "date range filter present")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(" -", f)
    sys.exit(1)
print("ALL SMOKE TESTS PASSED")
