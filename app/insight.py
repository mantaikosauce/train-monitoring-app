"""Cross-subsystem insight for the summary dashboard: zones, line paths, ranked
events and trends. Pure functions over session results, no Streamlit calls, so
tests/test_insight.py can exercise them directly.
"""
from __future__ import annotations

import pandas as pd
import pydeck as pdk

from livemap import BRANCHES, LINES, STATE_RGB, TYPE_COLOR

# Singapore planning regions, approximated from station centroids. Good enough to
# say "north" or "east"; not a URA boundary file.
ZONES = {
    "Central":    {"rgb": [98, 84, 214],  "hint": "CBD, Orchard, Toa Payoh, Bishan"},
    "North":      {"rgb": [31, 143, 92],  "hint": "Woodlands, Yishun, Sembawang"},
    "North-East": {"rgb": [214, 138, 0],  "hint": "Serangoon, Hougang, Sengkang, Punggol"},
    "East":       {"rgb": [0, 149, 236],  "hint": "Bedok, Tampines, Pasir Ris, Changi"},
    "West":       {"rgb": [214, 40, 40],  "hint": "Jurong, Clementi, Bukit Batok, Tuas"},
}


def zone_of(lon: float, lat: float) -> str:
    if lon < 103.76:
        return "West"
    if lon > 103.93 or (lon > 103.90 and lat < 1.37):
        return "East"
    if lat > 1.405:
        return "North"
    if lon > 103.86 and lat > 1.33:
        return "North-East"
    if lon >= 103.76 and lon <= 103.79 and lat < 1.36:
        return "West"
    return "Central"


def with_zones(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["zone"] = [zone_of(a, b) for a, b in zip(out["lon"], out["lat"])]
    return out


def zone_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for z in ZONES:
        d = df[df["zone"] == z]
        lines = sorted({code for s in d["lines"] for code in s.split(", ") if code != "—"})
        rows.append({"zone": z, "stations": int(len(d)), "underground": int((d["level"] == "Underground").sum()),
                     "lines": ", ".join(lines) or "—", "hint": ZONES[z]["hint"]})
    return pd.DataFrame(rows)


def zones_for_line(df: pd.DataFrame, line: str | None) -> list[str]:
    if not line or line not in LINES:
        return []
    sel = df[df["key"].isin(LINES[line][1])]
    return [z for z in ZONES if z in set(sel["zone"])]


def line_paths(df: pd.DataFrame) -> list[dict]:
    """Ordered station centroids per line, for a PathLayer. Missing stations are skipped."""
    lookup = df.drop_duplicates("key").set_index("key")
    paths = []
    for code, (hexcol, names) in LINES.items():
        rgb = [int(hexcol[i:i + 2], 16) for i in (1, 3, 5)]
        for seq in BRANCHES.get(code, [names]):
            pts = [[float(lookup.loc[n, "lon"]), float(lookup.loc[n, "lat"])] for n in seq if n in lookup.index]
            if len(pts) >= 2:
                paths.append({"line": code, "color": rgb, "path": pts, "n": len(pts)})
    return paths


def weather_layers(weather: dict | None) -> list:
    """Rain gauges as blue discs sized by the last 5-minute rainfall, temperature as text."""
    if not weather or not weather.get("ok") or weather.get("stations") is None or weather["stations"].empty:
        return []
    w = weather["stations"].copy()
    layers = []
    if "rain_mm" in w:
        r = w.dropna(subset=["rain_mm"]).copy()
        r["radius"] = 250 + 400 * r["rain_mm"].clip(0, 10)
        r["color"] = [[0, 120, 220, 140] if v > 0 else [0, 120, 220, 40] for v in r["rain_mm"]]
        r["label"] = [f"{s} · rain {v:.1f} mm" for s, v in zip(r["station"], r["rain_mm"])]
        layers.append(pdk.Layer("ScatterplotLayer", data=r, get_position="[lon, lat]", get_fill_color="color",
                                get_radius="radius", pickable=True, stroked=False))
    if "temp_c" in w:
        t = w.dropna(subset=["temp_c"]).copy()
        t["text"] = [f"{v:.1f}°" for v in t["temp_c"]]
        layers.append(pdk.Layer("TextLayer", data=t, get_position="[lon, lat]", get_text="text",
                                get_size=13, get_color=[40, 60, 90], get_text_anchor='"start"',
                                get_alignment_baseline='"bottom"', get_pixel_offset=[6, -6]))
    return layers


def network_deck(df: pd.DataFrame, line: str | None, state: str, mode: str = "lines",
                 weather: dict | None = None, view: pdk.ViewState | None = None,
                 focus: list[str] | None = None) -> pdk.Deck:
    """Detailed map: line paths in official colours, stations coloured by zone or type,
    the chosen line lifted with a halo in the verdict colour. With `focus`, every other
    line and its stations are dimmed so the eye lands on the lines that matter."""
    d = df.copy()
    if mode == "zones":
        d["color"] = [ZONES[z]["rgb"] for z in d["zone"]]
    else:
        d["color"] = [TYPE_COLOR.get(t, TYPE_COLOR["MRT"]) for t in d["type"]]
    if focus:
        on = {n for ln in focus for n in LINES.get(ln, ("", []))[1]}
        d["color"] = [c if k in on else [90, 100, 112, 60] for c, k in zip(d["color"], d["key"])]
    layers = []
    paths = line_paths(d)
    if paths:
        if focus:
            dim = [p for p in paths if p["line"] not in focus]
            lit = [p for p in paths if p["line"] in focus]
            if dim:
                layers.append(pdk.Layer("PathLayer", data=dim, get_path="path", get_color="color",
                                        width_min_pixels=2, get_width=40, opacity=0.12, pickable=False))
            if lit:
                layers.append(pdk.Layer("PathLayer", data=lit, get_path="path", get_color="color",
                                        width_min_pixels=4, get_width=80, opacity=0.95, pickable=True))
        else:
            layers.append(pdk.Layer("PathLayer", data=paths, get_path="path", get_color="color",
                                    width_min_pixels=3, get_width=60, opacity=0.9 if mode == "lines" else 0.35,
                                    pickable=True))
    layers.append(pdk.Layer("ScatterplotLayer", data=d, get_position="[lon, lat]", get_fill_color="color",
                            get_radius=130, pickable=True, stroked=True, get_line_color=[255, 255, 255],
                            line_width_min_pixels=1, opacity=0.95))
    view = pdk.ViewState(latitude=1.352, longitude=103.82, zoom=10.7, pitch=0)
    if line in LINES:
        sel = d[d["key"].isin(LINES[line][1])].copy()
        if not sel.empty:
            sel["halo"] = [STATE_RGB.get(state, STATE_RGB["unknown"])] * len(sel)
            layers.insert(0, pdk.Layer("ScatterplotLayer", data=sel, get_position="[lon, lat]",
                                       get_fill_color="halo", get_radius=520, opacity=0.3, stroked=False))
    layers = weather_layers(weather) + layers
    if view is not None:
        view = view  # explicit camera wins over the line-centred default
        globals()["_last_view"] = view
    tooltip = {"html": "<b>{name}{label}</b><br/>{zone} {type} {level}<br/>{lines}",
               "style": {"backgroundColor": "#10151c", "color": "#f7f8fa", "fontSize": "12px"}}
    return pdk.Deck(layers=layers, initial_view_state=view, tooltip=tooltip,
                    map_provider="carto", map_style=__import__("theme").map_style())


# ------------------------------------------------------------------ events

SHM_WATCH, SHM_ALERT = 0.5, 0.8


def events(results: dict) -> list[dict]:
    """Every noteworthy finding across subsystems, ranked by severity in [0, 1]."""
    ev: list[dict] = []
    door = results.get("door")
    if door is not None and not door["cycles"].empty:
        c = door["cycles"]
        ab = c[c["prediction"] == "Abnormal resistance"]
        for r in ab.itertuples():
            ev.append({"severity": 0.55 + 0.4 * float(r.p_abnormal), "state": "alert", "subsystem": "Door",
                       "title": f"Door cycle {r.cycle} ({r.operation}) against abnormal resistance",
                       "detail": f"{r.cur_mean_mid:.0f} mA sustained · P {r.p_abnormal:.0%} · {r.start_time}",
                       "page": "door"})
    shm = results.get("shm")
    if shm:
        for f in shm["files"]:
            d = float(f["damage"])
            if d >= SHM_WATCH:
                ev.append({"severity": min(1.0, 0.5 + 0.5 * d), "state": "alert" if d >= SHM_ALERT else "watch",
                           "subsystem": "Structural health",
                           "title": f"{f['file_id']} has used {d:.0%} of its fatigue life",
                           "detail": f"D = {d:.3f} · {f['n_cycles']:,} cycles · top 0.1% of cycles do {f['top_share']:.0%} of the damage",
                           "page": "shm"})
    rail = results.get("rail")
    if rail:
        for f in rail["files"]:
            if f["prediction"] != "Normal":
                p = f["proba"][f["prediction"]]
                ev.append({"severity": 0.6 + 0.4 * p, "state": "alert", "subsystem": "Rail corrugation",
                           "title": f"{f['file_id']}: {f['prediction']} rail corrugated",
                           "detail": f"P {p:.0%} · {f['speed']['speed_km_h']:.0f} km/h · side RMS I {f['side_i_rms']:.2f} / II {f['side_ii_rms']:.2f}",
                           "page": "rail"})
    acv = results.get("acv")
    if acv:
        for f in acv["files"]:
            top, second = f["ranking"][0], f["ranking"][1]
            pm = f.get("peer_mean", f["scores"])
            s1, s2 = pm.get(top), pm.get(second)
            gap = (s1 - s2) if (s1 is not None and s2 is not None) else 0.0
            ev.append({"severity": 0.45 + min(0.4, 4 * max(gap, 0.0)), "state": "alert" if gap >= 0.03 else "watch",
                       "subsystem": "Air conditioning",
                       "title": f"Car {top} most likely refrigerant leak in {f['file_id']}",
                       "detail": f"{s1:+.2f} °C over the other cars · margin {gap:.2f} °C to Car {second}",
                       "page": "acv"})
    return sorted(ev, key=lambda e: -e["severity"])


def top_events(ev: list[dict], n: int = 3) -> list[dict]:
    """The headline list: the worst event of each subsystem first, so three door
    cycles cannot crowd out a cracked structure; then the remaining by severity."""
    seen, first, rest = set(), [], []
    for e in ev:
        (rest if e["subsystem"] in seen else first).append(e)
        seen.add(e["subsystem"])
    return (first + rest)[:n]


# ------------------------------------------------------------------ trends

def trends(results: dict) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    door = results.get("door")
    if door is not None and not door["cycles"].empty:
        c = door["cycles"].copy()
        c["window_min"] = (c["start_s"] // 300 * 5).astype(int)
        g = c.groupby("window_min").agg(cycles=("cycle", "size"),
                                        abnormal=("prediction", lambda s: int((s == "Abnormal resistance").sum())),
                                        mean_current=("cur_mean_mid", "mean")).reset_index()
        g["abnormal_rate"] = g["abnormal"] / g["cycles"]
        out["door"] = g
    shm = results.get("shm")
    if shm:
        d = pd.DataFrame([{"file_id": f["file_id"], "damage": f["damage"], "n_cycles": f["n_cycles"],
                           "max_range": f["max_range"]} for f in shm["files"]])
        out["shm"] = d.sort_values("damage", ascending=False).reset_index(drop=True)
    rail = results.get("rail")
    if rail:
        rows = []
        for f in rail["files"]:
            num = "".join(ch for ch in f["file_id"] if ch.isdigit())
            rows.append({"file_id": f["file_id"], "order": int(num) if num else 0,
                         "p_corrugated": 1 - f["proba"]["Normal"], "prediction": f["prediction"],
                         "speed_km_h": f["speed"]["speed_km_h"]})
        out["rail"] = pd.DataFrame(rows).sort_values("order").reset_index(drop=True)
    acv = results.get("acv")
    if acv:
        f = acv["files"][0]
        ex = f["excess_timeline"].copy()
        top = f["ranking"][0]
        if top in ex.columns and ex["time"].notna().any():
            ex["day"] = ex["time"].dt.floor("D")
            daily = ex.groupby("day")[top].mean().reset_index().rename(columns={top: "excess"})
            daily["car"] = top
            out["acv"] = daily
    return out


def fleet_state(results: dict) -> dict[str, str]:
    """Worst state per subsystem key, 'unknown' when not run."""
    st = {k: "unknown" for k in ("door", "shm", "rail", "acv")}
    door = results.get("door")
    if door is not None and not door["cycles"].empty:
        st["door"] = "alert" if (door["cycles"]["prediction"] == "Abnormal resistance").any() else "ok"
    if results.get("shm"):
        worst = max(f["damage"] for f in results["shm"]["files"])
        st["shm"] = "alert" if worst >= SHM_ALERT else "watch" if worst >= SHM_WATCH else "ok"
    if results.get("rail"):
        st["rail"] = "alert" if (results["rail"]["predictions"]["prediction"] != "Normal").any() else "ok"
    if results.get("acv"):
        st["acv"] = "watch"
        for e in events({"acv": results["acv"]}):
            if e["state"] == "alert":
                st["acv"] = "alert"
    return st


# ------------------------------------------------------------------ event log on the map

def station_events(ev: pd.DataFrame, stations_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate logged events per station: count, worst state, faults, latest time."""
    if ev.empty or stations_df.empty:
        return pd.DataFrame(columns=["station", "lat", "lon", "n", "faults", "worst", "last"])
    rank = {"alert": 3, "watch": 2, "ok": 1, "unknown": 0}
    e = ev.dropna(subset=["station"]).copy()
    if e.empty:
        return pd.DataFrame(columns=["station", "lat", "lon", "n", "faults", "worst", "last"])
    lookup = stations_df.drop_duplicates("key").set_index("key")
    e["key"] = e["station"].astype(str).str.upper()
    e = e[e["key"].isin(lookup.index)]
    g = e.groupby("key").agg(n=("state", "size"),
                             faults=("state", lambda s: int((s == "alert").sum())),
                             worst=("state", lambda s: max(s, key=lambda x: rank.get(x, 0))),
                             last=("time", "max")).reset_index()
    g["station"] = [lookup.loc[k, "name"] for k in g["key"]]
    g["lat"] = [float(lookup.loc[k, "lat"]) for k in g["key"]]
    g["lon"] = [float(lookup.loc[k, "lon"]) for k in g["key"]]
    return g


def event_layers(agg: pd.DataFrame) -> list:
    if agg.empty:
        return []
    d = agg.copy()
    d["color"] = [STATE_RGB.get(w, STATE_RGB["unknown"]) + [200] for w in d["worst"]]
    d["radius"] = 220 + 60 * d["n"].clip(0, 20)
    d["label"] = [f"{s}: {n} event(s), {f} fault(s), latest {pd.Timestamp(t).strftime('%d %b %H:%M')}"
                  for s, n, f, t in zip(d["station"], d["n"], d["faults"], d["last"])]
    d["text"] = d["faults"].astype(str)
    return [pdk.Layer("ScatterplotLayer", data=d, get_position="[lon, lat]", get_fill_color="color",
                      get_radius="radius", pickable=True, stroked=True, get_line_color=[255, 255, 255, 180],
                      line_width_min_pixels=1),
            pdk.Layer("TextLayer", data=d[d["faults"] > 0], get_position="[lon, lat]", get_text="text",
                      get_size=12, get_color=[255, 255, 255], get_text_anchor='"middle"',
                      get_alignment_baseline='"center"')]


def fleet_deck(df: pd.DataFrame, agg: pd.DataFrame, lines: list[str], weather: dict | None = None,
               view: pdk.ViewState | None = None) -> pdk.Deck:
    """All line paths drawn; the selected lines bright, the rest dimmed; event markers per
    station; weather underneath."""
    d = df.copy()
    d["color"] = [TYPE_COLOR.get(t, TYPE_COLOR["MRT"]) for t in d["type"]]
    layers = weather_layers(weather)
    paths = line_paths(d)
    lit = [p for p in paths if not lines or p["line"] in lines]
    dim = [p for p in paths if lines and p["line"] not in lines]
    if dim:
        layers.append(pdk.Layer("PathLayer", data=dim, get_path="path", get_color="color",
                                width_min_pixels=2, get_width=40, opacity=0.12))
    if lit:
        layers.append(pdk.Layer("PathLayer", data=lit, get_path="path", get_color="color",
                                width_min_pixels=3, get_width=60, opacity=0.85))
    d["label"] = ""
    layers.append(pdk.Layer("ScatterplotLayer", data=d, get_position="[lon, lat]", get_fill_color="color",
                            get_radius=90, pickable=True, opacity=0.7))
    layers += event_layers(agg)
    if view is None:
        view = view_for(agg if not agg.empty else None)
    tooltip = {"html": "<b>{name}{station}</b><br/>{label}{zone} {type}<br/>{lines}",
               "style": {"backgroundColor": "#10151c", "color": "#f7f8fa", "fontSize": "12px"}}
    return pdk.Deck(layers=layers, initial_view_state=view, tooltip=tooltip,
                    map_provider="carto", map_style=__import__("theme").map_style())


def crowd_layer(crowd: dict | None, stations_df: pd.DataFrame) -> list:
    """Platform crowd density as coloured rings around stations (DataMall PCDRealTime)."""
    from livemap import CROWD_RGB, CROWD_WORD
    if not crowd or not crowd.get("ok") or crowd["rows"].empty or stations_df.empty:
        return []
    lookup = stations_df.drop_duplicates("key").set_index("key")
    d = crowd["rows"][crowd["rows"]["name"].isin(lookup.index)].copy()
    if d.empty:
        return []
    d["lat"] = [float(lookup.loc[n, "lat"]) for n in d["name"]]
    d["lon"] = [float(lookup.loc[n, "lon"]) for n in d["name"]]
    d["color"] = [CROWD_RGB.get(lv, [120, 120, 120]) + [90] for lv in d["level"]]
    d["radius"] = [320 if lv == "h" else 240 if lv == "m" else 170 for lv in d["level"]]
    d["label"] = [f"{c} crowd {CROWD_WORD.get(lv, '?')}" for c, lv in zip(d["code"], d["level"])]
    d["name"] = d["name"].str.title()
    return [pdk.Layer("ScatterplotLayer", data=d, get_position="[lon, lat]", get_fill_color="color",
                      get_radius="radius", pickable=True, stroked=True, get_line_color=[255, 255, 255, 120],
                      line_width_min_pixels=1)]


def alert_layer(alerts: dict | None, stations_df: pd.DataFrame) -> list:
    """Stations named in live TrainServiceAlerts segments, as red rings."""
    if not alerts or not alerts.get("ok") or stations_df.empty:
        return []
    names = {n for seg in alerts.get("segments", []) for n in seg.get("StationNames", [])}
    if not names:
        return []
    d = stations_df[stations_df["key"].isin(names)].copy()
    d["label"] = " · service disruption"
    return [pdk.Layer("ScatterplotLayer", data=d, get_position="[lon, lat]", get_fill_color=[224, 67, 79, 60],
                      get_radius=650, pickable=True, stroked=True, get_line_color=[224, 67, 79, 220],
                      line_width_min_pixels=2)]


# ------------------------------------------------------------------ fleet health index

WEIGHT = {"alert": 1.0, "watch": 0.4, "ok": 0.0, "unknown": 0.0}


def health_index(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per train: a 0-100 health index and the worst state per subsystem.
    Index = 100 minus severity-weighted faults, saturating at 40 so one bad train
    still sorts below a train with two watches; the shape follows fleet-health
    products (a single score per asset, drill-down by subsystem)."""
    cols = ["train", "line", "health", "events", "faults", "door", "shm", "rail", "acv", "last"]
    if ev.empty or ev["train"].notna().sum() == 0:
        return pd.DataFrame(columns=cols)
    e = ev.dropna(subset=["train"]).copy()
    rank = {"alert": 3, "watch": 2, "ok": 1, "unknown": 0}
    rows = []
    for train, g in e.groupby("train"):
        penalty = sum(WEIGHT[s] * (10 + 20 * float(sv)) for s, sv in zip(g["state"], g["severity"]))
        r = {"train": train, "line": g["line"].iloc[0], "health": max(40.0, 100.0 - penalty),
             "events": int(len(g)), "faults": int((g["state"] == "alert").sum()), "last": g["time"].max()}
        for k in ("door", "shm", "rail", "acv"):
            sub = g[g["subsystem"] == k]
            r[k] = max(sub["state"], key=lambda x: rank.get(x, 0)) if len(sub) else "unknown"
        rows.append(r)
    return pd.DataFrame(rows, columns=cols).sort_values(["health", "last"], ascending=[True, False]).reset_index(drop=True)


# ------------------------------------------------------------------ camera

SINGAPORE = dict(latitude=1.352, longitude=103.82, zoom=10.6)


def view_for(points: pd.DataFrame | None, duration_ms: int = 900) -> pdk.ViewState:
    """Camera that fits the given points (lat/lon columns); whole island when empty.
    deck.gl animates between view states when transition_duration is set."""
    if points is None or len(points) == 0:
        v = dict(SINGAPORE)
    else:
        lat0, lat1 = float(points["lat"].min()), float(points["lat"].max())
        lon0, lon1 = float(points["lon"].min()), float(points["lon"].max())
        span = max(lat1 - lat0, (lon1 - lon0) * 0.98, 0.01)
        zoom = max(9.5, min(13.5, 11.3 - 2.2 * (span / 0.25 - 1)))   # ~island span 0.25° -> 11.3
        v = dict(latitude=(lat0 + lat1) / 2, longitude=(lon0 + lon1) / 2, zoom=zoom)
    # transition_duration animates between view states in deck.gl; an interpolator given as a
    # plain string is not understood by the JSON renderer and can blank the whole map, so none is set.
    return pdk.ViewState(**v, pitch=0, bearing=0, transition_duration=duration_ms)


def affected_lines(ev: pd.DataFrame) -> list[str]:
    """Lines that carry at least one fault or watch in the given events."""
    if ev is None or ev.empty or "line" not in ev:
        return []
    bad = ev[ev["state"].isin(["alert", "watch"])].dropna(subset=["line"])
    return [ln for ln in LINES if ln in set(bad["line"])]


def stations_on(df: pd.DataFrame, lines: list[str]) -> pd.DataFrame:
    names = {n for ln in lines for n in LINES.get(ln, ("", []))[1]}
    return df[df["key"].isin(names)]
