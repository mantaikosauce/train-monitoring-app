"""Data-driven schematics, drawn only from values the models actually produce.

Each function returns inline SVG for st.markdown(unsafe_allow_html=True). Nothing
here invents a position or a value: the ACV train colours cars by their own
ranking scores, the Rail train colours each axle box by its own vibration RMS,
the Door diagram annotates a real cycle. SHM has no location data, so it gets
no schematic (see shm page: damage gauges per file instead).
"""
from __future__ import annotations

from html import escape

import pandas as pd

RAMP = ["#bfe0f7", "#7cc0ea", "#0095ec", "#d68a00", "#b3121f"]   # cool -> hot, on white
GLYPH = {"alert": "✕", "watch": "◐", "ok": "●", "unknown": "—"}      # the console's one severity alphabet


def _ramp(v: float) -> str:
    v = 0.0 if v != v else max(0.0, min(1.0, v))
    return RAMP[min(4, int(v * 5))]


def _car(x: float, y: float, w: float, h: float, fill: str, label: str, sub: str = "", stroke: str = "#10151c") -> str:
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-opacity=".25"/>'
            f'<text x="{x + w / 2}" y="{y + 22}" text-anchor="middle" font-size="13" font-weight="600" fill="#10151c">{escape(label)}</text>'
            + (f'<text x="{x + w / 2}" y="{y + 40}" text-anchor="middle" font-size="11" fill="#10151c" fill-opacity=".85">{escape(sub)}</text>' if sub else ""))


def acv_train(ranking: list[str], scores: dict, hot: dict | None, unobserved: list[str]) -> str:
    """Eight cars in physical order (01..08), coloured by rank: rank 1 red, then the ramp
    down. Each car shows its mean excess and, when the v2 rule applies, hot-episode share."""
    cars = sorted(ranking)
    n = len(cars)
    w, gap, h = 96, 8, 64
    total = n * w + (n - 1) * gap + 40
    out = [f'<svg viewBox="0 0 {total} 120" width="100%" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Train cars ranked by likelihood of refrigerant leak">']
    out.append('<text x="20" y="16" font-size="11" fill="#626c7a" letter-spacing="1.5">TRAIN · CARS 01–08 · RANK 1 = MOST LIKELY LEAK</text>')
    for i, c in enumerate(cars):
        x = 20 + i * (w + gap)
        rank = ranking.index(c) + 1
        if c in unobserved:
            fill, sub = "#ffffff", "no evidence"
        else:
            fill = "#e0434f" if rank == 1 else _ramp(1 - (rank - 1) / max(1, n - 1)) if rank <= 3 else "#dfe6ee"
            s = scores.get(c)
            sub = (f"{hot.get(c, 0):.1%} hot · " if hot else "") + (f"{s:+.2f} °C" if isinstance(s, (int, float)) else "")
        out.append(_car(x, 30, w, h, fill, f"Car {c} · #{rank}", sub))
        # airflow: three waves above the roof unit; the suspect car's air is warm and sluggish
        warm = rank == 1 and c not in unobserved
        col = "#e0434f" if warm else "#2e8fd4"
        dur = "2.6s" if warm else "1.3s"
        for k in range(3):
            wx = x + 20 + k * 26
            out.append(f'<path d="M{wx} 26 q4 -5 8 0 t8 0" fill="none" stroke="{col}" stroke-width="2" opacity=".0">'
                       f'<animate attributeName="opacity" values="0;.9;0" dur="{dur}" begin="{k * 0.3}s" repeatCount="indefinite"/>'
                       f'<animateTransform attributeName="transform" type="translate" values="0 4;0 -6" dur="{dur}" begin="{k * 0.3}s" repeatCount="indefinite"/></path>')
        out.append(f'<text x="{x + w - 10}" y="44" text-anchor="end" font-size="13" fill="{"#e0434f" if rank == 1 else "#626c7a"}">{GLYPH["alert"] if rank == 1 else GLYPH["ok"]}</text>')
        # bogies
        for bx in (x + 18, x + w - 18):
            out.append(f'<circle cx="{bx}" cy="104" r="6" fill="#eef1f5" stroke="#626c7a"/>')
    out.append('</svg>')
    return "".join(out)


def rail_train(grid: pd.DataFrame, prediction: str) -> str:
    """Side view: 8 cars, each with 4 axle boxes per rail side, coloured by that box's
    own vibration RMS (normalised within the recording). Side I row above, Side II below."""
    cars = sorted(grid["car"].unique())
    w, gap = 104, 8
    total = 20 + len(cars) * (w + gap) + 60
    lookup = {(int(r.car), int(r.position)): (float(r.value), float(r.rms)) for r in grid.itertuples()}
    out = [f'<svg viewBox="0 0 {total} 170" width="100%" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Axle-box vibration per car and rail side">']
    hot = "SIDE I" if prediction == "Side I" else "SIDE II" if prediction == "Side II" else "NEITHER"
    out.append(f'<text x="20" y="16" font-size="11" fill="#626c7a" letter-spacing="1.5">AXLE-BOX VIBRATION · VERDICT {escape(prediction.upper())} · CORRUGATED RAIL: {hot}</text>')
    out.append('<text x="20" y="62" font-size="10" fill="#454f5c">Side I rail</text>')
    out.append('<text x="20" y="152" font-size="10" fill="#454f5c">Side II rail</text>')
    for i, car in enumerate(cars):
        x = 80 + i * (w + gap)
        out.append(f'<rect x="{x}" y="70" width="{w}" height="50" rx="8" fill="#eef1f5" stroke="#10151c" stroke-opacity=".2"/>')
        out.append(f'<text x="{x + w / 2}" y="100" text-anchor="middle" font-size="12" font-weight="600" fill="#10151c">Car {car}</text>')
        for k, pos in enumerate((1, 3, 5, 7)):        # Side I boxes above
            v, rms = lookup.get((car, pos), (0.0, 0.0))
            out.append(f'<circle cx="{x + 14 + k * 25}" cy="60" r="8" fill="{_ramp(v)}" stroke="#ffffff"><title>Car {car} pos {pos} · RMS {rms:.3f}</title></circle>')
        for k, pos in enumerate((2, 4, 6, 8)):        # Side II boxes below
            v, rms = lookup.get((car, pos), (0.0, 0.0))
            out.append(f'<circle cx="{x + 14 + k * 25}" cy="132" r="8" fill="{_ramp(v)}" stroke="#ffffff"><title>Car {car} pos {pos} · RMS {rms:.3f}</title></circle>')
    # rails
    for y, side in ((48, "Side I"), (144, "Side II")):
        hit = prediction == side
        col = "#e0434f" if hit else "#b8c2cc"
        out.append(f'<line x1="70" y1="{y}" x2="{total - 20}" y2="{y}" stroke="{col}" stroke-width="{4 if hit else 2}" stroke-dasharray="{"6 4" if hit else "0"}">'
                   + ('<animate attributeName="stroke-dashoffset" values="0;-20" dur="1s" repeatCount="indefinite"/>'
                      '<animate attributeName="opacity" values="1;.45;1" dur="1.6s" repeatCount="indefinite"/>' if hit else "") + '</line>')
        out.append(f'<text x="{total - 12}" y="{y + 5}" text-anchor="end" font-size="13" fill="{col if hit else "#16a97a"}">{GLYPH["alert"] if hit else GLYPH["ok"]}</text>')
    out.append('</svg>')
    return "".join(out)


def door_cycle(row: pd.Series) -> str:
    """One door cycle: leaf travel, sustained current, duration, verdict, from that cycle's values."""
    abnormal = row["prediction"] == "Abnormal resistance"
    col = "#e0434f" if abnormal else "#16a97a"
    travel = float(row.get("pos_range", 0) or 0)
    out = ['<svg viewBox="0 0 640 150" width="100%" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Door cycle diagram">',
           '<text x="16" y="16" font-size="11" fill="#626c7a" letter-spacing="1.5">DOOR LEAF · MOTOR · ONE CYCLE</text>',
           # door frame and leaf
           '<rect x="16" y="30" width="220" height="100" rx="6" fill="#f7f8fa" stroke="#b8c2cc"/>',
           f'<rect x="{30 if row["operation"] == "Close" else 120}" y="38" width="100" height="84" rx="4" fill="{col}" fill-opacity=".85">'
           f'<animate attributeName="x" values="{"120;30;30" if row["operation"] == "Close" else "30;120;120"}" dur="{max(1.2, float(row["duration_s"]) * 0.6):.1f}s" repeatCount="indefinite"/></rect>',
           f'<text x="220" y="52" text-anchor="end" font-size="14" fill="{col}">{GLYPH["alert" if abnormal else "ok"]}</text>',
           f'<text x="126" y="140" text-anchor="middle" font-size="10" fill="#454f5c">leaf {row["operation"].lower()}s · travel {travel:.0f}</text>',
           # motor + values
           '<circle cx="300" cy="80" r="26" fill="#eef1f5" stroke="#626c7a"/>',
           '<text x="300" y="84" text-anchor="middle" font-size="10" fill="#10151c">motor</text>',
           f'<line x1="236" y1="80" x2="274" y2="80" stroke="{col}" stroke-width="3">'
           + ('<animate attributeName="opacity" values="1;.3;1" dur="0.8s" repeatCount="indefinite"/>' if abnormal else "") + '</line>',
           f'<text x="350" y="56" font-size="13" font-weight="600" fill="#10151c">{escape(str(row["prediction"]))}</text>',
           f'<text x="350" y="78" font-size="12" fill="#454f5c">sustained current {row["cur_mean_mid"]:.0f} mA · P(abnormal) {row["p_abnormal"]:.0%}</text>',
           f'<text x="350" y="98" font-size="12" fill="#454f5c">duration {row["duration_s"]:.2f} s · current per back-EMF {row.get("cur_per_emf", float("nan")):.2f}</text>',
           f'<text x="350" y="118" font-size="11" fill="#626c7a">{escape(str(row["start_time"]))}</text>',
           '</svg>']
    return "".join(out)


def shm_gauge(file_id: str, damage: float, segments_left: float) -> str:
    """An arc from 0 to 1 with the needle at D. Watch band 0.5, alert 0.8, failure at 1."""
    import math
    d = max(0.0, min(1.0, damage))
    state = "alert" if d >= 0.8 else "watch" if d >= 0.5 else "ok"
    col = {"alert": "#e0434f", "watch": "#d68a00", "ok": "#16a97a"}[state]

    def arc(a0, a1, color, width=14):
        r, cx, cy = 90, 130, 120
        x0, y0 = cx + r * math.cos(math.pi * (1 - a0)), cy - r * math.sin(math.pi * (1 - a0))
        x1, y1 = cx + r * math.cos(math.pi * (1 - a1)), cy - r * math.sin(math.pi * (1 - a1))
        return f'<path d="M{x0:.1f} {y0:.1f} A{r} {r} 0 0 1 {x1:.1f} {y1:.1f}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="butt"/>'
    ang = math.pi * (1 - d)
    nx, ny = 130 + 78 * math.cos(ang), 120 - 78 * math.sin(ang)
    return ("".join([
        '<svg viewBox="0 0 260 150" width="100%" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Fatigue damage gauge">',
        arc(0.0, 0.5, "#16a97a"), arc(0.5, 0.8, "#d68a00"), arc(0.8, 1.0, "#e0434f"),
        f'<line x1="130" y1="120" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#10151c" stroke-width="3" stroke-linecap="round">'
        f'<animate attributeName="x2" from="40" to="{nx:.1f}" dur="1s" fill="freeze"/><animate attributeName="y2" from="120" to="{ny:.1f}" dur="1s" fill="freeze"/></line>',
        '<circle cx="130" cy="120" r="5" fill="#10151c"/>',
        f'<text x="130" y="100" text-anchor="middle" font-size="22" font-weight="600" fill="{col}">{d:.2f} {GLYPH[state]}</text>',
        f'<text x="130" y="142" text-anchor="middle" font-size="11" fill="#626c7a">{escape(file_id)} · {d:.0%} of fatigue life · {segments_left:.1f} segments left at this rate</text>',
        '<text x="34" y="140" font-size="10" fill="#626c7a">0</text><text x="220" y="140" font-size="10" fill="#626c7a">D = 1</text>',
        '</svg>']))
