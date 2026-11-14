#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
space_health_monitor.py  (v2)
Astronot sağlığı + kabin telemetrisi için kural tabanlı, trend duyarlı, kişisel-baz çizgili sınıflandırıcı.

Komutlar:
  classify girdi.csv -o sonuc.csv            5 dk'lık telemetri (kabin + fizyoloji) -> seviye, alan (domain), öneri
  daily    gunluk.csv --telemetry sonuc.csv  günlük check-in (uyku, ruh hali, stres, direnç egzersizi, baş ağrısı)
  limits   girdi.csv                          veriden alt/üst limit tahmini (yalnızca bilgi amaçlı)

Challenge'ın dört risk alanı + ortam:
  environment (CO2, O2, basınç, ısı, nem) · radiation (doz hızı + kümülatif doz) · cardio (nabız, SpO2, HRV, tansiyon...)
  behavioral (uyku, ruh hali, stres) · bone (direnç egzersizi = kemik/kas karşı-önlemi) · immune (VERİ YOK)
NOT: Limitlerin yalnızca [NASA] işaretlileri NASA dokümanlarına dayanır; geri kalanı mühendislik tahminidir.
"""
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

LEVELS = ["NORMAL", "CAUTION", "WARNING", "CRITICAL"]


@dataclass(frozen=True)
class Limit:
    """low  = (kritik, uyarı, dikkat): değer bu eşiklerin ALTINA düşerse seviye artar
       high = (dikkat, uyarı, kritik): değer bu eşiklerin ÜSTÜNE çıkarsa seviye artar"""
    low: Optional[Tuple[float, float, float]] = None
    high: Optional[Tuple[float, float, float]] = None
    unit: str = ""


# ---------------------------------------------------------------------------
# 1) LİMİTLER
# ---------------------------------------------------------------------------
LIMITS_REST = {
    # [NASA] NASA-STD-3001 Vol.2 [V2 6004]: 1 saatlik ort. pCO2 <= 3.0 mmHg. ISS işletme limiti ~4.0 mmHg (2010'dan beri);
    #        mürettebat belirtileri ~2.3 mmHg'den itibaren bildirdi. 5.0 = SMAC 5.3 (7-180 gün) altında temkinli tahmin.
    "pco2": Limit(high=(3.0, 4.0, 5.0), unit="mmHg"),
    # [NASA] normoksi PIO2 145-155 mmHg, hafif hipoksi alt limiti 127 mmHg; PIO2=(PB-47)*FIO2 -> 760 mmHg'de 127 ~ kabin ppO2 135.
    "po2": Limit(low=(135, 140, 145), high=(165, 175, 190), unit="mmHg"),
    "pressure": Limit(low=(96, 99, 100), high=(103, 105, 110), unit="kPa"),          # tahmin (nominal 101.3)
    # Radyasyon doz HIZI (tahmin). [NASA] referans: GCR tasarım tavanı 1.3 mSv/gün (=1300 uSv/gün) serbest uzayda.
    "radiation": Limit(high=(450, 600, 1000), unit="uSv/gun"),
    "cabin_temp": Limit(low=(10, 16, 18), high=(27, 30, 35), unit="C"),
    "humidity": Limit(low=(15, 25, 30), high=(60, 70, 85), unit="%"),
    "spo2": Limit(low=(92, 94, 95), unit="%"),
    "hr": Limit(low=(40, 45, 50), high=(100, 120, 140), unit="bpm"),
    "hrv": Limit(low=(15, 20, 30), unit="ms"),
    "sys": Limit(low=(80, 90, 100), high=(130, 140, 160), unit="mmHg"),
    "dia": Limit(low=(50, 55, 60), high=(85, 90, 100), unit="mmHg"),
    "core_temp": Limit(low=(35.0, 36.0, 36.4), high=(37.5, 38.0, 39.0), unit="C"),
    "resp": Limit(low=(8, 10, 12), high=(20, 24, 30), unit="/dk"),
}
LIMITS_EXERTION = {
    "hr": Limit(low=(40, 45, 50), high=(155, 170, 185), unit="bpm"),
    "resp": Limit(low=(8, 10, 12), high=(30, 38, 45), unit="/dk"),
    "sys": Limit(low=(80, 90, 100), high=(160, 180, 200), unit="mmHg"),
    "dia": Limit(low=(50, 55, 60), high=(90, 100, 110), unit="mmHg"),
    "core_temp": Limit(low=(35.0, 36.0, 36.4), high=(38.3, 39.0, 40.0), unit="C"),
    "hrv": Limit(unit="ms"),
}
LIMITS_SLEEP = {
    "hr": Limit(low=(35, 40, 45), high=(100, 120, 140), unit="bpm"),
    "resp": Limit(low=(6, 8, 10), high=(20, 24, 30), unit="/dk"),
}
# Günlük check-in limitleri (tahmin; kişisel baz çizgisi ve seri kuralı ile birlikte okunmalı)
LIMITS_DAILY = {
    "sleep": Limit(low=(4, 5, 6), unit="saat"),
    "mood": Limit(low=(0.5, 1.5, 2.5), unit="1-5"),
    "stress": Limit(high=(3.5, 4.5, 6), unit="1-5"),
    "resist": Limit(low=(0, 10, 20), unit="dk"),
}
CAREER_DOSE_LIMIT_MSV = 600.0   # [NASA] NASA-STD-3001 Vol.1 [V1 4030]: kariyer etkin doz < 600 mSv
SPE_EVENT_LIMIT_MSV = 250.0     # [NASA] NASA-STD-3001 Vol.1 [V1 4031]: tasarım referans SPE < 250 mSv

COLS = {
    "pco2": "cabin_pco2_mmhg", "po2": "cabin_po2_mmhg", "pressure": "cabin_pressure_kpa",
    "radiation": "radiation_rate_usv_day", "cabin_temp": "cabin_temp_c",
    "humidity": "relative_humidity_percent",
    "spo2": "spo2_percent", "hr": "heart_rate_bpm", "hrv": "hrv_rmssd_ms",
    "sys": "systolic_bp_mmhg", "dia": "diastolic_bp_mmhg",
    "core_temp": "core_temp_c", "resp": "respiration_rate_bpm",
}
DOMAINS = {
    "environment": ["pco2", "po2", "pressure", "cabin_temp", "humidity"],
    "radiation": ["radiation"],
    "cardio": ["spo2", "hr", "hrv", "sys", "dia", "core_temp", "resp"],
}
ENV = DOMAINS["environment"] + DOMAINS["radiation"]
BODY = DOMAINS["cardio"]

PLAUSIBLE = {
    "pco2": (0, 50), "po2": (0, 400), "pressure": (20, 150), "radiation": (0, 1e5),
    "cabin_temp": (-20, 80), "humidity": (0, 100), "spo2": (50, 100), "hr": (20, 250),
    "hrv": (0, 300), "sys": (50, 260), "dia": (30, 160), "core_temp": (30, 43), "resp": (2, 70),
}

TREND_PCO2_RISE = 0.5       # mmHg/15dk  -> en az WARNING
TREND_P_DROP_WARN = 1.0     # kPa/15dk   -> en az WARNING (sızıntı şüphesi)
TREND_P_DROP_CRIT = 3.0     # kPa/15dk   -> CRITICAL
EXERT_HR_MIN = 100
SLEEP_HR_MAX = 60
SLEEP_RR_MAX = 13
COMPOUND_N = 3
# Kişisel baz çizgisi: ilk 24 saatteki sakin satırlardan medyan/MAD
BASELINE_HOURS, BASELINE_MIN_ROWS, PERS_Z, PERS_MIN_IND = 24, 12, 4.0, 2
BASE_COLS = {"hr": ("heart_rate_bpm", +1, 2.0), "hrv": ("hrv_rmssd_ms", -1, 4.0),
             "spo2": ("spo2_percent", -1, 0.4), "resp": ("respiration_rate_bpm", +1, 0.8)}

# (parametre, yön) -> (İngilizce, Türkçe)
ACTIONS = {
    ("pco2", "HIGH"): ("Check CO2 scrubber/CDRA, increase ventilation, reduce exertion",
                       "CO2 scrubber/CDRA'yı kontrol et, havalandırmayı artır, eforu azalt"),
    ("po2", "LOW"): ("Check O2 supply and look for a leak", "O2 beslemesini kontrol et, sızıntı olasılığını değerlendir"),
    ("po2", "HIGH"): ("O2 excess is a fire risk: reduce O2 input", "O2 fazlası yangın riski: O2 girişini azalt"),
    ("pressure", "LOW"): ("Locate leak, isolate the module, prepare emergency O2",
                          "Sızıntıyı bul, bölmeyi izole et, acil O2 hazırla"),
    ("radiation", "HIGH"): ("Possible solar particle event: move to the shielded area",
                            "Olası güneş parçacık olayı: korunaklı bölmeye geç"),
    ("spo2", "LOW"): ("Check cabin gases first (CO2/O2/pressure), then the crew member",
                      "Önce kabin gazlarını (CO2/O2/basınç), sonra astronotu kontrol et"),
    ("hrv", "LOW"): ("Possible stress, fatigue or gas effect: keep monitoring, review workload",
                     "Olası stres, yorgunluk veya gaz etkisi: izlemeye devam et, iş yükünü gözden geçir"),
    ("core_temp", "HIGH"): ("Heat stress: increase cooling, rest", "Isı stresi: soğutmayı artır, dinlen"),
    ("hr", "HIGH"): ("Separate exertion, stress and hypoxia as causes", "Efor, stres ve hipoksi ayrımını yap"),
    ("resp", "HIGH"): ("Check for CO2/hypoxia, reduce exertion", "CO2/hipoksi kontrol et, eforu azalt"),
    ("sys", "HIGH"): ("Rest and re-measure; check workload and sleep", "Dinlen ve yeniden ölç; iş yükü ve uykuyu gözden geçir"),
    ("dia", "HIGH"): ("Rest and re-measure; check workload and sleep", "Dinlen ve yeniden ölç; iş yükü ve uykuyu gözden geçir"),
    ("sleep", "LOW"): ("Protect sleep window, reduce evening workload, discuss with flight surgeon",
                       "Uyku saatini koru, akşam iş yükünü azalt, uçuş cerrahıyla görüş"),
    ("mood", "LOW"): ("Schedule a private call / peer check-in; review isolation stressors",
                      "Özel görüşme / ekip arkadaşıyla check-in planla; izolasyon stres kaynaklarını gözden geçir"),
    ("stress", "HIGH"): ("Reduce task load, use relaxation and recreation time", "Görev yükünü azalt, dinlenme ve rekreasyon zamanı kullan"),
    ("resist", "LOW"): ("Complete the missed resistive-exercise session (bone/muscle countermeasure)",
                        "Kaçırılan direnç egzersizini tamamla (kemik/kas karşı-önlemi)"),
    ("headache", "HIGH"): ("Check cabin CO2; headache can be a CO2 symptom", "Kabin CO2'yu kontrol et; baş ağrısı CO2 belirtisi olabilir"),
    ("persdev", "HIGH"): ("Compare with own baseline; check sleep, workload, cabin air", "Kendi baz çizginle karşılaştır; uyku, iş yükü ve kabin havasını kontrol et"),
}
WHY = {
    ("pco2", "HIGH"): ("High CO2 causes headache and reduced performance; ISS crews report symptoms from ~2.3 mmHg.",
                       "Yüksek CO2 baş ağrısı ve performans düşüşü yapar; ISS mürettebatı ~2.3 mmHg'den itibaren belirti bildiriyor."),
    ("po2", "LOW"): ("Low O2 partial pressure leads to hypoxia.", "Düşük O2 kısmi basıncı hipoksiye yol açar."),
    ("pressure", "LOW"): ("Falling pressure can mean a leak; O2 availability falls with it.",
                          "Basınç düşüşü sızıntı belirtisi olabilir; O2 de birlikte düşer."),
    ("radiation", "HIGH"): ("Radiation dose adds to the career limit (600 mSv) and can cause acute effects in a solar particle event.",
                            "Radyasyon dozu kariyer limitine (600 mSv) eklenir; güneş parçacık olayında akut etki yapabilir."),
    ("spo2", "LOW"): ("Low blood oxygen can come from cabin gas problems or breathing problems.",
                      "Düşük kan oksijeni kabin gazı veya solunum sorunlarından kaynaklanabilir."),
    ("hrv", "LOW"): ("Low HRV reflects physiological stress or fatigue.", "Düşük HRV fizyolojik stres veya yorgunluğu yansıtır."),
    ("sleep", "LOW"): ("Sleep loss hurts cognition and mood and compounds other risks.",
                       "Uyku kaybı bilişi ve ruh halini bozar, diğer riskleri artırır."),
    ("mood", "LOW"): ("Isolation and confinement can degrade behavioral health.", "İzolasyon ve kapalı ortam davranışsal sağlığı bozabilir."),
    ("stress", "HIGH"): ("Sustained stress affects sleep, performance and the immune system.",
                         "Süregelen stres uykuyu, performansı ve bağışıklığı etkiler."),
    ("resist", "LOW"): ("Skipping resistive exercise accelerates bone and muscle loss in microgravity.",
                        "Direnç egzersizini atlamak mikro-yerçekiminde kemik ve kas kaybını hızlandırır."),
    ("headache", "HIGH"): ("Headache together with raised cabin CO2 is a recognised CO2 symptom.",
                           "Yüksek kabin CO2 ile birlikte baş ağrısı bilinen bir CO2 belirtisidir."),
    ("persdev", "HIGH"): ("Several indicators moved away from this astronaut's own normal at the same time.",
                          "Birden çok gösterge astronotun kendi normalinden aynı anda uzaklaştı."),
}


def classify_value(v, lim):
    """(yön, seviye) döndürür. yön: LOW / HIGH / OK"""
    if lim.low:
        crit, warn, caut = lim.low
        if v < crit: return "LOW", 3
        if v < warn: return "LOW", 2
        if v < caut: return "LOW", 1
    if lim.high:
        caut, warn, crit = lim.high
        if v > crit: return "HIGH", 3
        if v > warn: return "HIGH", 2
        if v > caut: return "HIGH", 1
    return "OK", 0


def _rate15(cur, prev, dt_min):
    if dt_min is None or not (0 < dt_min <= 60):
        return None
    return (cur - prev) * 15.0 / dt_min


def _texts(keys):
    keys = [k for k in keys if k in ACTIONS][:2]
    return (" ; ".join(ACTIONS[k][0] for k in keys), " ; ".join(ACTIONS[k][1] for k in keys),
            " ".join(WHY[k][0] for k in keys if k in WHY), " ".join(WHY[k][1] for k in keys if k in WHY))


# ---------------------------------------------------------------------------
# 2) TELEMETRİ SINIFLANDIRMA
# ---------------------------------------------------------------------------
def _classify_one(g):
    recs = []
    prev_vals, prev_t, prev_ctx, prev_level = {}, None, "REST", 0
    cum_msv, last_rate = 0.0, None

    for _, r in g.iterrows():
        t = r["timestamp"]
        dt = (t - prev_t).total_seconds() / 60 if prev_t is not None else None

        vals, invalid = {}, []
        for k, col in COLS.items():
            v = r.get(col, np.nan)
            if pd.isna(v):
                continue
            lo, hi = PLAUSIBLE[k]
            if lo <= v <= hi:
                vals[k] = float(v)
            else:
                invalid.append(k)

        env = {k: classify_value(vals[k], LIMITS_REST[k]) for k in ENV if k in vals}
        env_max = max((lv for _, lv in env.values()), default=0)

        trend_lvl, notes, trend_keys = 0, [], []

        def rate(k):
            return _rate15(vals[k], prev_vals[k], dt) if (k in vals and k in prev_vals) else None

        d = rate("pco2")
        if d is not None and d >= TREND_PCO2_RISE:
            trend_lvl = max(trend_lvl, 2)
            notes.append(f"pCO2 rising fast (+{d:.2f} mmHg/15min)")
            trend_keys.append(("pco2", "HIGH"))
        d = rate("pressure")
        if d is not None and d <= -TREND_P_DROP_WARN:
            trend_lvl = max(trend_lvl, 3 if d <= -TREND_P_DROP_CRIT else 2)
            notes.append(f"pressure dropping ({d:.2f} kPa/15min)")
            trend_keys.append(("pressure", "LOW"))

        exertion = (vals.get("hr", 0) >= EXERT_HR_MIN and vals.get("spo2", 0) >= 95
                    and env_max <= 1 and trend_lvl < 2)
        sleeping = (not exertion and vals.get("hr", 999) < SLEEP_HR_MAX and vals.get("resp", 999) < SLEEP_RR_MAX
                    and vals.get("spo2", 0) >= 94 and env_max <= 1 and trend_lvl < 2)
        if exertion:
            ctx, lim = "EXERCISE", {**LIMITS_REST, **LIMITS_EXERTION}
        elif sleeping:
            ctx, lim = "SLEEP", {**LIMITS_REST, **LIMITS_SLEEP}
        else:
            ctx, lim = ("RECOVERY" if prev_ctx == "EXERCISE" else "REST"), LIMITS_REST

        body = {k: classify_value(vals[k], lim[k]) for k in BODY if k in vals}
        res = {**env, **body}

        dom = {n: max([res[k][1] for k in keys if k in res], default=0) for n, keys in DOMAINS.items()}
        dom["environment"] = max(dom["environment"], trend_lvl)

        overall = max(max((lv for _, lv in res.values()), default=0), trend_lvl)
        n_flag = sum(1 for _, lv in res.values() if lv >= 1)
        if n_flag >= COMPOUND_N and overall < 2:
            overall = 2
            notes.append(f"{n_flag} sensors at their limit at once (compound risk)")

        flags = [f"{k}:{d_}:{LEVELS[lv]}={vals[k]:g}{LIMITS_REST[k].unit}"
                 for k, (d_, lv) in sorted(res.items(), key=lambda x: -x[1][1]) if lv >= 1]
        keys = list(trend_keys)
        for k, (d_, lv) in res.items():
            if lv == overall and lv >= 1 and (k, d_) not in keys:
                keys.append((k, d_))
        a_en, a_tr, w_en, w_tr = _texts(keys)

        if "radiation" in vals:
            last_rate = vals["radiation"]
        if dt is not None and 0 < dt <= 60 and last_rate is not None:
            cum_msv += last_rate * (dt / 1440.0) / 1000.0   # uSv/gün -> mSv

        rec = {
            "pred_status": LEVELS[overall], "pred_level": overall, "pred_context": ctx,
            "pred_recovering": bool(prev_level >= 2 and overall < prev_level),
            "pred_flags": " | ".join(flags), "pred_notes": " | ".join(notes),
            "pred_action": a_en, "pred_action_tr": a_tr, "pred_why": w_en, "pred_why_tr": w_tr,
            "pred_invalid_sensors": ",".join(invalid),
            "dom_environment": dom["environment"], "dom_radiation": dom["radiation"], "dom_cardio": dom["cardio"],
            "cum_dose_msv": round(cum_msv, 5),
            "cum_dose_pct_career": round(100 * cum_msv / CAREER_DOSE_LIMIT_MSV, 5),
        }
        for k in COLS:
            rec[f"st_{k}"] = "NA" if k not in vals else (
                "OK" if res[k][0] == "OK" else f"{res[k][0]}/{LEVELS[res[k][1]]}")
        recs.append(rec)
        prev_vals, prev_t, prev_ctx, prev_level = vals, t, ctx, overall

    out = pd.concat([g.reset_index(drop=True), pd.DataFrame(recs)], axis=1)
    return _personal_baseline(out)


def _personal_baseline(res):
    """Astronotun KENDİ normalinden sapma: >=PERS_MIN_IND gösterge |z|>=PERS_Z (kötü yönde) ise CAUTION."""
    t0 = res["timestamp"].min()
    calm = res["pred_context"].isin(["REST", "SLEEP"])
    in_base = (res["timestamp"] < t0 + pd.Timedelta(hours=BASELINE_HOURS)) & (res["pred_level"] == 0) & calm
    z = pd.DataFrame(np.nan, index=res.index, columns=list(BASE_COLS))
    for ctx in ("REST", "SLEEP"):
        m_ctx = res["pred_context"] == ctx
        b = res[in_base & m_ctx]
        if len(b) < BASELINE_MIN_ROWS:
            continue
        for k, (col, sign, floor) in BASE_COLS.items():
            lo, hi = PLAUSIBLE[k]
            x = pd.to_numeric(res[col], errors="coerce")
            x = x.where((x >= lo) & (x <= hi))
            xb = x[b.index].dropna()
            if len(xb) < BASELINE_MIN_ROWS:
                continue
            med = xb.median()
            scale = max(1.4826 * (xb - med).abs().median(), floor)
            z.loc[m_ctx, k] = (sign * (x - med) / scale)[m_ctx]
    cnt = (z >= PERS_Z).sum(axis=1)
    res["pers_dev_count"] = cnt
    cand = (cnt >= PERS_MIN_IND) & (res["pred_level"] == 0) & calm
    if cand.any():
        a_en, a_tr, w_en, w_tr = _texts([("persdev", "HIGH")])
        res.loc[cand, "pred_level"] = 1
        res.loc[cand, "pred_status"] = "CAUTION"
        res.loc[cand, "dom_cardio"] = np.maximum(res.loc[cand, "dom_cardio"], 1)
        res.loc[cand, "pred_notes"] = "deviates from personal baseline (" + cnt[cand].astype(str) + " indicators)"
        res.loc[cand, "pred_action"], res.loc[cand, "pred_action_tr"] = a_en, a_tr
        res.loc[cand, "pred_why"], res.loc[cand, "pred_why_tr"] = w_en, w_tr
    return res


def classify_dataframe(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["astronaut_id", "timestamp"]).reset_index(drop=True)
    parts = [_classify_one(g) for _, g in df.groupby("astronaut_id", sort=False)]
    return pd.concat(parts).sort_values(["timestamp", "astronaut_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3) GÜNLÜK CHECK-IN (davranışsal sağlık + kemik/kas karşı-önlemi + baş ağrısı)
# ---------------------------------------------------------------------------
DAILY_COLS = {"sleep": "sleep_hours", "mood": "mood_1_5", "stress": "stress_1_5", "resist": "resistive_exercise_min"}


def classify_daily(daily, telemetry=None):
    """daily: date, astronaut_id, sleep_hours, mood_1_5, stress_1_5, resistive_exercise_min, headache(0/1).
    telemetry (opsiyonel): günlük maksimum pCO2'yi bulmak için (baş ağrısı + CO2 ilişkisi)."""
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.date
    d = d.sort_values(["astronaut_id", "date"]).reset_index(drop=True)
    d["max_pco2"] = np.nan
    if telemetry is not None:
        t = telemetry.copy()
        t["timestamp"] = pd.to_datetime(t["timestamp"], utc=True)
        p = pd.to_numeric(t["cabin_pco2_mmhg"], errors="coerce")
        p = p.where((p >= PLAUSIBLE["pco2"][0]) & (p <= PLAUSIBLE["pco2"][1]))
        mx = p.groupby(t["timestamp"].dt.date).max()
        d["max_pco2"] = d["date"].map(mx)

    for k, col in DAILY_COLS.items():
        d[f"lvl_{k}"] = [classify_value(v, LIMITS_DAILY[k])[1] if pd.notna(v) else 0 for v in d[col]]
        # 3 gün üst üste sınırda -> bir seviye yukarı (kişinin kendi değişimini yakalar)
        s = d.groupby("astronaut_id")[f"lvl_{k}"]
        streak = (d[f"lvl_{k}"] >= 1) & (s.shift(1) >= 1) & (s.shift(2) >= 1)
        d[f"streak_{k}"] = streak
        d.loc[streak, f"lvl_{k}"] = np.minimum(d.loc[streak, f"lvl_{k}"] + 1, 3)

    head = d["headache"].fillna(0).astype(int) == 1
    d["lvl_headache"] = np.where(head, np.where(d["max_pco2"] > 3.0, 2, 1), 0)

    # davranışsal birleşik kural: uyku/ruh hali/stres'ten >=2'si sınırda -> en az WARNING
    beh = d[["lvl_sleep", "lvl_mood", "lvl_stress"]]
    d["dom_behavioral"] = np.where((beh >= 1).sum(axis=1) >= 2, np.maximum(beh.max(axis=1), 2), beh.max(axis=1))
    d["dom_bone"] = d["lvl_resist"]
    d["daily_level"] = d[["dom_behavioral", "dom_bone", "lvl_headache"]].max(axis=1).astype(int)
    d["daily_status"] = d["daily_level"].map(dict(enumerate(LEVELS)))

    notes, keys_col = [], []
    for _, r in d.iterrows():
        n, ks = [], []
        for k, direction in (("sleep", "LOW"), ("mood", "LOW"), ("stress", "HIGH"), ("resist", "LOW")):
            if r[f"lvl_{k}"] >= 1:
                n.append(f"{k}:{LEVELS[int(r[f'lvl_{k}'])]}" + (" (3-day streak)" if r[f"streak_{k}"] else ""))
                ks.append((k, direction))
        if r["lvl_headache"] >= 1:
            n.append("headache" + (f" with max pCO2 {r['max_pco2']:.1f} mmHg" if pd.notna(r["max_pco2"]) else ""))
            ks.append(("headache", "HIGH"))
        notes.append(" | ".join(n))
        keys_col.append(ks)
    d["daily_flags"] = notes
    txt = [_texts(ks) for ks in keys_col]
    d["daily_action"] = [x[0] for x in txt]
    d["daily_action_tr"] = [x[1] for x in txt]
    d["daily_why"] = [x[2] for x in txt]
    d["daily_why_tr"] = [x[3] for x in txt]
    return d


# ---------------------------------------------------------------------------
# 4) DEĞERLENDİRME + LİMİT TAHMİNİ
# ---------------------------------------------------------------------------
LABEL_LEVEL = {"NOMINAL": 0, "EXERCISE": 0, "RECOVERY": 0, "RECOVERING": 1, "WARNING": 2, "CRITICAL": 3}


def evaluate(df, show=15):
    if "risk_status" not in df.columns:
        return
    y = df["risk_status"].map(LABEL_LEVEL)
    ok = y.notna()
    y, p = y[ok].astype(int), df.loc[ok, "pred_level"]
    diff = (p - y).abs()
    print(f"\n=== Degerlendirme ({len(y)} satir) ===")
    print(f"Birebir uyum        : {(diff == 0).mean():.1%}")
    print(f"+-1 seviye icinde   : {(diff <= 1).mean():.1%}")
    crit = y == 3
    if crit.any():
        print(f"CRITICAL yakalama   : {(p[crit] == 3).mean():.1%} (>=WARNING: {(p[crit] >= 2).mean():.1%})")
    calm = y == 0
    if calm.any():
        print(f"Yanlis alarm (>=WARNING, etiket sakin): {(p[calm] >= 2).mean():.1%}")
    cm = pd.crosstab(y.map(dict(enumerate(LEVELS))).rename("etiket"),
                     p.map(dict(enumerate(LEVELS))).rename("tahmin"))
    print("\nKarisiklik matrisi:\n", cm)
    bad = df[ok][diff >= 1]
    if len(bad):
        print(f"\nUyusmayan satirlar (ilk {show}):")
        print(bad[["timestamp", "astronaut_id", "risk_status", "pred_status", "pred_flags"]].head(show).to_string(index=False))


def estimate_limits(df, baseline_status=("NOMINAL",), k=3.0):
    """Sakin satırlardan ortalama ± k·std ve yüzdelik limitler (yalnızca bilgi; sabit NASA limitlerinin yerine geçmez)."""
    base = df[df["risk_status"].isin(baseline_status)] if "risk_status" in df.columns else df
    rows = []
    for key, col in COLS.items():
        if col not in base.columns:
            continue
        s = pd.to_numeric(base[col], errors="coerce").dropna()
        lo_p, hi_p = PLAUSIBLE[key]
        s = s[(s >= lo_p) & (s <= hi_p)]
        if len(s) < 2:
            continue
        ref = LIMITS_REST[key]
        rows.append({
            "parameter": key, "unit": ref.unit, "n": len(s), "mean": s.mean(), "std": s.std(),
            f"est_low(mean-{k:g}sd)": s.mean() - k * s.std(), f"est_high(mean+{k:g}sd)": s.mean() + k * s.std(),
            "p01": s.quantile(0.01), "p99": s.quantile(0.99),
            "ref_caution_low": ref.low[2] if ref.low else np.nan,
            "ref_caution_high": ref.high[0] if ref.high else np.nan,
        })
    return pd.DataFrame(rows).round(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("classify")
    c.add_argument("input"); c.add_argument("-o", "--output", default="classified.csv")
    c.add_argument("--no-eval", action="store_true")
    dd = sub.add_parser("daily")
    dd.add_argument("input"); dd.add_argument("--telemetry"); dd.add_argument("-o", "--output", default="daily_classified.csv")
    l = sub.add_parser("limits")
    l.add_argument("input"); l.add_argument("-o", "--output", default="estimated_limits.csv")
    l.add_argument("--baseline", nargs="+", default=["NOMINAL"]); l.add_argument("-k", type=float, default=3.0)
    a = ap.parse_args()

    df = pd.read_csv(a.input)
    if a.cmd == "classify":
        out = classify_dataframe(df)
        out.to_csv(a.output, index=False)
        print(f"{len(out)} satir siniflandi -> {a.output}")
        print(out["pred_status"].value_counts().reindex(LEVELS).fillna(0).astype(int).to_string())
        if not a.no_eval:
            evaluate(out)
    elif a.cmd == "daily":
        tele = pd.read_csv(a.telemetry) if a.telemetry else None
        out = classify_daily(df, tele)
        out.to_csv(a.output, index=False)
        print(f"{len(out)} gunluk kayit -> {a.output}")
        print(out[["date", "astronaut_id", "daily_status", "daily_flags"]].to_string(index=False))
    else:
        res = estimate_limits(df, tuple(a.baseline), a.k)
        res.to_csv(a.output, index=False)
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
