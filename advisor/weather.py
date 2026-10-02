"""Weather access (Open-Meteo). Fully generic: which variables to fetch, how to aggregate them and
which time windows exist all come from policies/weather.yaml. Any failure raises WeatherError."""
from __future__ import annotations

import datetime as dt

import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 10
AGG = {"max": max, "min": min, "sum": sum}


class WeatherError(Exception):
    """Any failure to obtain trustworthy location or weather data."""


def geocode(name: str) -> dict:
    try:
        r = requests.get(GEOCODE_URL, params={"name": name, "count": 5, "language": "en", "format": "json"},
                         timeout=TIMEOUT)
        r.raise_for_status()
        results = r.json().get("results") or []
    except (requests.RequestException, ValueError) as e:
        raise WeatherError(f"the location service could not be reached ({type(e).__name__})") from e
    if not results:
        raise WeatherError(f"I couldn't find a place called '{name}'")
    top = results[0]  # first candidate; n_candidates is surfaced so ambiguity isn't hidden
    parts = [top.get("name"), top.get("admin1"), top.get("country")]
    return {"display": ", ".join(p for p in parts if p), "latitude": top["latitude"],
            "longitude": top["longitude"], "n_candidates": len(results)}


def fetch_raw(latitude: float, longitude: float, variables) -> dict:
    params = {"latitude": latitude, "longitude": longitude, "hourly": ",".join(sorted(variables)),
              "current": "temperature_2m", "timezone": "auto", "forecast_days": 3}
    try:
        r = requests.get(FORECAST_URL, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError) as e:
        raise WeatherError(f"the weather service could not be reached ({type(e).__name__})") from e
    if "hourly" not in data or "current" not in data:
        raise WeatherError("the weather service returned an unexpected response")
    return data


def build_facts(raw: dict, fields: dict, windows: dict, window_name: str):
    """Deterministically turns raw hourly data into the facts SOPs are checked against.
    Every number the bot may later quote is produced here, nowhere else."""
    w = windows[window_name]
    now = dt.datetime.fromisoformat(raw["current"]["time"])  # local time (timezone=auto)
    day = (now + dt.timedelta(days=w["day_offset"])).date()
    start = now.hour if w["start"] == "now" else int(w["start"])
    end = now.hour if w["end"] == "now" else int(w["end"])
    if w["day_offset"] == 0:
        if end < now.hour:
            raise WeatherError("that time window has already passed today")
        start = max(start, now.hour)
    times = [dt.datetime.fromisoformat(t) for t in raw["hourly"]["time"]]
    facts = {}
    for name, f in fields.items():
        series = raw["hourly"].get(f["var"])
        if series is None:
            facts[name] = None
            continue
        lo, hi = (0, 23) if f.get("scope") == "day" else (start, end)
        if f.get("hours"):
            lo, hi = max(lo, f["hours"][0]), min(hi, f["hours"][1])
        vals = [series[i] for i, t in enumerate(times)
                if t.date() == day and lo <= t.hour <= hi and series[i] is not None]
        facts[name] = round(AGG[f["agg"]](vals), 1) if vals else None
    if all(v is None for v in facts.values()):
        raise WeatherError("the weather service had no data for that time window")
    meta = {"window": window_name, "date": day.isoformat(), "hours": f"{start:02d}:00-{end:02d}:59",
            "fetched_at": raw["current"]["time"]}
    return facts, meta
