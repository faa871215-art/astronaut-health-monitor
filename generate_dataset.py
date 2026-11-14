#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_dataset.py
Tahmini (sentetik) astronot + kabin telemetri veri seti uretir; sema, yuklenen CSV ile ayni.

Senaryolar: uyku, egzersiz (2x/gun), CO2 scrubber doygunlugu, gunes parcacik olayi (SPE), mikro sizinti.
Kabin degerleri tum murettebat icin ORTAK, fizyoloji kisiye ozgu. Gurultu + eksik veri + sensor ariza spike'lari icerir.

Etiketler (risk_status) olay evresinden uretilir (siddet zarfi e: >=0.8 CRITICAL, >=0.3 WARNING,
inis evresinde e<0.3 RECOVERING). Siniflandirici bu etiketleri gormez.

  python generate_dataset.py --crew 4 --days 3 --seed 42 -o synthetic_telemetry.csv
"""
import argparse
import numpy as np
import pandas as pd

STEP_MIN = 5
SPD = 24 * 60 // STEP_MIN  # gunluk adim (288)

LOGS = {
    "BASELINE": "Stable telemetry", "SLEEP": "Sleep period", "EXERCISE": "Cardio exercise session",
    "SCRUBBER_SATURATION": "CO2 scrubber saturation, pCO2 rising",
    "SOLAR_PARTICLE_EVENT": "SPE alert, radiation dose rate rising",
    "MICRO_LEAK": "Pressure falling, micro-leak suspected",
}


def trapezoid(n, start, rise, hold, fall):
    """0->1 yukselis, 1 plato, 1->0 inis. (zarf, inis_maskesi) dondurur."""
    i = np.arange(n)
    e = np.zeros(n)
    up = (i >= start) & (i < start + rise)
    e[up] = (i[up] - start + 1) / rise
    h = (i >= start + rise) & (i < start + rise + hold)
    e[h] = 1.0
    s2 = start + rise + hold
    dn = (i >= s2) & (i < s2 + fall)
    e[dn] = 1 - (i[dn] - s2 + 1) / fall
    return e, (i >= s2) & (i < s2 + fall)


def ar1(rng, n, sigma, phi=0.9):
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, sigma * np.sqrt(1 - phi ** 2))
    return x


def build(crew, days, seed):
    rng = np.random.default_rng(seed)
    n = SPD * days
    idx = np.arange(n)
    ts = pd.Timestamp("2026-10-03T08:00:00Z") + pd.to_timedelta(idx * STEP_MIN, unit="m")
    day_step = idx % SPD
    sleep = ((day_step >= 168) & (day_step < 264)).astype(float)  # 22:00-06:00

    # ---- Ortak kabin olaylari (her gun, birbirini kesmeyecek sekilde) ----
    e_co2, e_rad, e_leak = np.zeros(n), np.zeros(n), np.zeros(n)
    fall = np.zeros(n, bool)
    which = np.array(["BASELINE"] * n, dtype=object)
    peaks = {}
    for d in range(days):
        o = d * SPD
        for name, arr, base, rise, hold, dn in [
            ("SCRUBBER_SATURATION", e_co2, 36, 9, 3, 6),
            ("SOLAR_PARTICLE_EVENT", e_rad, 100, 4, 4, 6),
            ("MICRO_LEAK", e_leak, 150, 6, 3, 6),
        ]:
            st = o + base + int(rng.integers(-5, 6))
            e, f = trapezoid(n, st, rise, hold, dn)
            arr += e
            fall |= f
            which[(e > 0)] = name
        peaks[d] = (rng.uniform(5.3, 6.0), rng.uniform(1200, 1800), rng.uniform(92.0, 95.0))
    pk = lambda j: np.array([peaks[i // SPD][j] for i in idx])
    e_all = e_co2 + e_rad + e_leak

    # ---- Egzersiz zarflari (kisiye ozgu, 2 seans/gun) ----
    ex = np.zeros((crew, n))
    for c in range(crew):
        for d in range(days):
            for base in (12, 108):
                st = d * SPD + base + int(rng.integers(-3, 4))
                ex[c] += trapezoid(n, st, 2, 4, 3)[0]
    ex = np.clip(ex, 0, 1)
    crew_ex = ex.mean(axis=0)

    # ---- Kabin ----
    pressure = 101.3 + ar1(rng, n, 0.04) - e_leak * (101.3 - pk(2))
    po2 = 155.0 * pressure / 101.3 + ar1(rng, n, 0.4)
    pco2 = 2.2 + ar1(rng, n, 0.08) + 0.25 * crew_ex + e_co2 * (pk(0) - 2.2)
    radiation = 315 + ar1(rng, n, 6) + e_rad * (pk(1) - 315)
    cabin_temp = 21.4 + 0.5 * np.sin(2 * np.pi * idx / SPD) + 0.5 * crew_ex + ar1(rng, n, 0.1)
    humidity = 46 + 2 * crew_ex + ar1(rng, n, 0.6)

    co2x = np.maximum(pco2 - 3.0, 0)       # pCO2 fazlasi
    hyp = np.maximum(155 - po2, 0)         # O2 acigi

    # ---- Etiketler ----
    base_label = np.where(e_all >= 0.8, "CRITICAL",
                 np.where(e_all >= 0.3, "WARNING",
                 np.where(fall & (e_all > 0), "RECOVERING", "NOMINAL")))

    frames = []
    for c in range(crew):
        hr0, hrv0 = rng.normal(66, 5), rng.normal(55, 8)
        s0, d0 = rng.normal(118, 5), rng.normal(77, 3)
        t0, rr0, sp0 = rng.normal(36.85, 0.1), rng.normal(14.5, 1), rng.normal(98.0, 0.3)
        e = ex[c]
        hr = hr0 - 10 * sleep + rng.uniform(65, 85) * e + 3 * co2x + 0.8 * hyp + rng.normal(0, 1.8, n)
        rr = rr0 - 2 * sleep + 11 * e + 1.5 * co2x + 0.5 * hyp + rng.normal(0, 0.6, n)
        sysb = s0 - 8 * sleep + 22 * e + 4 * co2x + 0.3 * hyp + rng.normal(0, 2, n)
        diab = d0 - 6 * sleep + 4 * e + 2.5 * co2x + 0.2 * hyp + rng.normal(0, 1.5, n)
        temp = t0 - 0.3 * sleep + 0.7 * e + rng.normal(0, 0.04, n)
        hrv = (hrv0 + 15 * sleep - 30 * e) * np.clip(1 - 0.12 * co2x - 0.01 * hyp, 0.35, 1) + rng.normal(0, 3, n)
        spo2 = sp0 - 0.3 * sleep - 1.2 * e - 1.2 * co2x - 0.25 * hyp + rng.normal(0, 0.2, n)

        label = base_label.copy()
        label[(e > 0.35) & (label == "NOMINAL")] = "EXERCISE"
        scen = which.copy()
        scen[(scen == "BASELINE") & (e > 0.35)] = "EXERCISE"
        scen[(scen == "BASELINE") & (sleep > 0)] = "SLEEP"

        f = pd.DataFrame({
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "astronaut_id": f"ASTRO-{c + 1:02d}",
            "heart_rate_bpm": hr.round(0), "spo2_percent": np.minimum(spo2, 100).round(1),
            "hrv_rmssd_ms": np.maximum(hrv, 5).round(0), "systolic_bp_mmhg": sysb.round(0),
            "diastolic_bp_mmhg": diab.round(0), "core_temp_c": temp.round(2),
            "respiration_rate_bpm": rr.round(0),
            "cabin_pco2_mmhg": (pco2 + rng.normal(0, 0.03, n)).round(2),
            "cabin_po2_mmhg": (po2 + rng.normal(0, 0.2, n)).round(1),
            "cabin_pressure_kpa": (pressure + rng.normal(0, 0.01, n)).round(2),
            "radiation_rate_usv_day": (radiation + rng.normal(0, 2, n)).round(0),
            "cabin_temp_c": cabin_temp.round(1), "relative_humidity_percent": humidity.round(1),
            "risk_status": label, "telemetry_log": [LOGS[s] for s in scen], "scenario": scen,
        })
        frames.append(f)

    df = pd.concat(frames).sort_values(["timestamp", "astronaut_id"]).reset_index(drop=True)

    # ---- Eksik veri ve sensor ariza spike'lari ----
    sensors = [c for c in df.columns if c.endswith(("_bpm", "_percent", "_ms", "_mmhg", "_c", "_kpa", "_day"))]
    for col in sensors:
        m = rng.random(len(df)) < 0.002
        df.loc[m, col] = np.nan
    spike = {"spo2_percent": 0.0, "heart_rate_bpm": 300.0, "cabin_pco2_mmhg": 99.0}
    for col, val in spike.items():
        m = rng.random(len(df)) < 0.0007
        df.loc[m, col] = val
    return df


def build_daily(crew, days, seed, tele):
    """Gunluk check-in: uyku, ruh hali, stres, direnc egzersizi, bas agrisi.
    Hikaye: ASTRO-03 davranissal dususe girer (uyku/ruh hali/stres), ASTRO-02 son gunlerde direnc egzersizini birakir.
    Bas agrisi olasiligi, o gun kabin pCO2 yuksekse artar."""
    rng = np.random.default_rng(seed + 1)
    p = pd.to_numeric(tele["cabin_pco2_mmhg"], errors="coerce")
    p = p.where(p < 50)                                   # sensor arizasi spike'larini at
    mx = p.groupby(tele["timestamp"].str[:10]).max()
    rows = []
    for d in range(days):
        date = (pd.Timestamp("2026-10-03") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        frac = d / max(days - 1, 1)
        day_pco2 = mx.get(date, np.nan)
        for c in range(crew):
            sleep = rng.normal(7.0, 0.35)
            mood = rng.normal(4.0, 0.4)
            stress = rng.normal(2.0, 0.4)
            resist = rng.normal(40, 5)
            aerobic = rng.normal(35, 5)
            if c == 2:                                    # davranissal dusus
                sleep -= 2.0 * frac; mood -= 2.0 * frac; stress += 2.0 * frac
            if c == 1:                                    # karsi-onlem uyumu dususu
                resist *= (1 - 0.92 * frac ** 2)
            p_head = 0.6 if (pd.notna(day_pco2) and day_pco2 >= 5.0) else 0.05
            rows.append({
                "date": date, "astronaut_id": f"ASTRO-{c + 1:02d}",
                "sleep_hours": round(float(np.clip(sleep, 2, 10)), 1),
                "mood_1_5": int(np.clip(round(mood), 1, 5)),
                "stress_1_5": int(np.clip(round(stress), 1, 5)),
                "resistive_exercise_min": int(max(round(resist), 0)),
                "aerobic_exercise_min": int(max(round(aerobic), 0)),
                "headache": int(rng.random() < p_head),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crew", type=int, default=4)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-o", "--output", default="synthetic_telemetry.csv")
    ap.add_argument("--daily-output", default="daily_checkins.csv")
    a = ap.parse_args()
    d = build(a.crew, a.days, a.seed)
    d.to_csv(a.output, index=False)
    print(f"{len(d)} satir -> {a.output}")
    print(d["risk_status"].value_counts().to_string())
    dd = build_daily(a.crew, a.days, a.seed, d)
    dd.to_csv(a.daily_output, index=False)
    print(f"{len(dd)} gunluk kayit -> {a.daily_output}")
