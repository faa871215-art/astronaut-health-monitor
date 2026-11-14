# -*- coding: utf-8 -*-
"""
Astronaut Health & Cabin Monitor — Streamlit dashboard (Multi-language + Dark Mode + Crew Comparison)
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from space_health_monitor import LIMITS_REST, PLAUSIBLE, COLS  # noqa: E402
import real_data  # noqa: E402

DATA_T = ROOT / "data" / "full_classified.csv"
DATA_D = ROOT / "data" / "daily_classified.csv"
LEVEL_COLOR = ["#2e7d32", "#1976d2", "#f9a825", "#c62828"]
NA_COLOR = "#546e7a"

# --- ÇOKLU DİL DESTEĞİ (5 DİL) ---
I18N = {
    "en": {
        "title": "Astronaut Health & Cabin Telemetry", "side": "Control panel", "astro": "Astronaut",
        "time": "Time (default: latest record)", "win": "Chart window (hours)",
        "replay": "Move the time slider to replay a scenario like a live feed.",
        "lvl": ["NORMAL", "CAUTION (watch)", "WARNING", "CRITICAL"],
        "ctx": {"REST": "rest", "EXERCISE": "exercise", "SLEEP": "sleep", "RECOVERY": "recovery"},
        "recov": "↘ improving", "live": "Live status", "domains": "Risk domains",
        "dom": {"environment": "Cabin", "radiation": "Radiation", "cardio": "Cardio-respiratory",
                "behavioral": "Behavioral", "bone": "Bone / Muscle", "immune": "Immune"},
        "no_data": "no data", "immune_note": "needs lab data",
        "m_pco2": "pCO₂ (mmHg)", "m_p": "Pressure (kPa)", "m_o2": "pO₂ (mmHg)", "m_hr": "Heart rate (bpm)",
        "m_spo2": "SpO₂ (%)", "m_dose": "Cumulative dose (mSv)", "m_dose_pct": "of career limit",
        "c_pco2": "Cabin pCO₂", "c_p": "Cabin pressure", "c_o2": "Cabin pO₂", "c_hr": "heart rate",
        "c_rad": "Radiation rate", "daily": "Daily check-ins", "d_sleep": "Sleep (h)", "d_mood": "Mood (1-5)",
        "d_stress": "Stress (1-5)", "d_res": "Exercise (min)", "d_none": "No daily check-in yet.",
        "why": "Why it matters", "steps": "Suggested response steps", "flags": "Active flags", "trend": "Trend note",
        "ok": "All parameters nominal. No action needed.", "watch": "Values near limits; keep monitoring.",
        "disc": "⚠️ Prototype automatic suggestions; simulated data.",
        "tab1": "Monitor", "tab2": "Crew Comparison", "tab3": "Live NASA Data",
        "comp_h": "Multi-Crew Telemetry Comparison", "comp_p": "Compare all astronauts simultaneously over time.",
        "live_h": "Real NASA data", "live_p": "Optional panels pulling public NASA APIs.",
        "donki": "Fetch Solar Events (DONKI)", "osdr": "Search OSDR Studies",
        "term": "Search term", "none_ev": "No SEP events found.", "fail": "Could not reach NASA service:",
        "src": "Data & Sources", "src_md": "- Simulated crew telemetry & NASA-STD-3001 limits."
    },
    "tr": {
        "title": "Astronot Sağlığı & Kabin Telemetrisi", "side": "Kontrol Paneli", "astro": "Astronot",
        "time": "Zaman (varsayılan: son kayıt)", "win": "Grafik Penceresi (saat)",
        "replay": "Zaman kaydırıcısı senaryoyu canlı akış gibi simüle eder.",
        "lvl": ["NORMAL", "İZLEME (CAUTION)", "UYARI (WARNING)", "KRİTİK (CRITICAL)"],
        "ctx": {"REST": "dinlenme", "EXERCISE": "egzersiz", "SLEEP": "uyku", "RECOVERY": "toparlanma"},
        "recov": "↘ iyileşme eğiliminde", "live": "Canlı Durum", "domains": "Risk Alanları",
        "dom": {"environment": "Kabin", "radiation": "Radyasyon", "cardio": "Kalp-Solunum",
                "behavioral": "Davranışsal", "bone": "Kemik / Kas", "immune": "Bağışıklık"},
        "no_data": "veri yok", "immune_note": "lab verisi gerekir",
        "m_pco2": "pCO₂ (mmHg)", "m_p": "Basınç (kPa)", "m_o2": "pO₂ (mmHg)", "m_hr": "Nabız (bpm)",
        "m_spo2": "SpO₂ (%)", "m_dose": "Kümülatif Doz (mSv)", "m_dose_pct": "kariyer limitinin",
        "c_pco2": "Kabin pCO₂", "c_p": "Kabin Basıncı", "c_o2": "Kabin pO₂", "c_hr": "nabız",
        "c_rad": "Radyasyon Hızı", "daily": "Günlük Check-in'ler", "d_sleep": "Uyku (saat)", "d_mood": "Ruh Hali (1-5)",
        "d_stress": "Stres (1-5)", "d_res": "Egzersiz (dk)", "d_none": "Henüz günlük check-in yok.",
        "why": "Neden Önemli", "steps": "Önerilen Müdahale Adımları", "flags": "Aktif Bayraklar", "trend": "Trend Notu",
        "ok": "Tüm parametreler nominal. Müdahale gerekmiyor.", "watch": "Sınıra yakın değerler var; izleyin.",
        "disc": "⚠️ Prototip otomatik önerileridir; veriler simülasyondur.",
        "tab1": "İzleme", "tab2": "Mürettebat Karşılaştırma", "tab3": "Canlı NASA Verisi",
        "comp_h": "Tüm Mürettebat Telemetri Karşılaştırması", "comp_p": "Tüm astronotların durumunu aynı zaman diliminde kıyaslayın.",
        "live_h": "Gerçek NASA Verisi", "live_p": "Halka açık NASA API'lerinden veri çeken opsiyonel panel.",
        "donki": "Güneş Olaylarını Getir (DONKI)", "osdr": "OSDR Araştırmalarında Ara",
        "term": "Arama Terimi", "none_ev": "SEP olayı bulunamadı.", "fail": "NASA servisine ulaşılamadı:",
        "src": "Veri ve Kaynaklar", "src_md": "- Simüle mürettebat telemetrisi ve NASA-STD-3001 limitleri."
    },
    "de": {
        "title": "Astronauten-Gesundheit & Kabinentelemetrie", "side": "Systemsteuerung", "astro": "Astronaut",
        "time": "Zeit (Standard: letzter Datensatz)", "win": "Zeitfenster (Stunden)",
        "replay": "Zeitregler zum Abspielen des Szenarios.",
        "lvl": ["NORMAL", "VORSICHT", "WARNUNG", "KRITISCH"],
        "ctx": {"REST": "Ruhe", "EXERCISE": "Training", "SLEEP": "Schlaf", "RECOVERY": "Erholung"},
        "recov": "↘ verbessert sich", "live": "Live-Status", "domains": "Risikobereiche",
        "dom": {"environment": "Kabine", "radiation": "Strahlung", "cardio": "Kardio",
                "behavioral": "Verhalten", "bone": "Knochen / Muskel", "immune": "Immunsystem"},
        "no_data": "keine Daten", "immune_note": "Labordaten erforderlich",
        "m_pco2": "pCO₂ (mmHg)", "m_p": "Druck (kPa)", "m_o2": "pO₂ (mmHg)", "m_hr": "Puls (bpm)",
        "m_spo2": "SpO₂ (%)", "m_dose": "Gesamtdosis (mSv)", "m_dose_pct": "des Limits",
        "c_pco2": "Kabinen-pCO₂", "c_p": "Kabinedruck", "c_o2": "Kabinen-pO₂", "c_hr": "Puls",
        "c_rad": "Strahlungsrate", "daily": "Tägliche Check-ins", "d_sleep": "Schlaf (Std)", "d_mood": "Stimmung (1-5)",
        "d_stress": "Stress (1-5)", "d_res": "Training (Min)", "d_none": "Noch kein Check-in.",
        "why": "Warum das wichtig ist", "steps": "Empfohlene Schritte", "flags": "Aktive Flags", "trend": "Trend",
        "ok": "Alle Parameter nominal.", "watch": "Werte nahe der Grenzwerte.",
        "disc": "⚠️ Prototyp-Empfehlungen; simulierte Daten.",
        "tab1": "Monitor", "tab2": "Crew-Vergleich", "tab3": "Live NASA Daten",
        "comp_h": "Missionsvergleich der Crew", "comp_p": "Vergleichen Sie alle Astronauten.",
        "live_h": "Echte NASA-Daten", "live_p": "Optionale Panels für öffentliche NASA-APIs.",
        "donki": "Solarereignisse abrufen", "osdr": "OSDR Studien suchen",
        "term": "Suchbegriff", "none_ev": "Keine Ereignisse.", "fail": "Dienst nicht erreichbar:",
        "src": "Daten & Quellen", "src_md": "- Simulierte Telemetrie & NASA-Standards."
    },
    "fr": {
        "title": "Santé des Astronautes & Télémétrie", "side": "Panneau de contrôle", "astro": "Astronaute",
        "time": "Temps (par défaut: dernier)", "win": "Fenêtre (heures)",
        "replay": "Curseur temporel pour rejouer le scénario.",
        "lvl": ["NORMAL", "ATTENTION", "AVERTISSEMENT", "CRITIQUE"],
        "ctx": {"REST": "repos", "EXERCISE": "exercice", "SLEEP": "sommeil", "RECOVERY": "récupération"},
        "recov": "↘ en amélioration", "live": "Statut en direct", "domains": "Domaines de risque",
        "dom": {"environment": "Cabine", "radiation": "Radiation", "cardio": "Cardio-respiratoire",
                "behavioral": "Comportemental", "bone": "Os / Muscle", "immune": "Immunitaires"},
        "no_data": "pas de données", "immune_note": "nécessite des analyses",
        "m_pco2": "pCO₂ (mmHg)", "m_p": "Pression (kPa)", "m_o2": "pO₂ (mmHg)", "m_hr": "Fréquence (bpm)",
        "m_spo2": "SpO₂ (%)", "m_dose": "Dose cumulée (mSv)", "m_dose_pct": "de la limite",
        "c_pco2": "pCO₂ cabine", "c_p": "Pression cabine", "c_o2": "pO₂ cabine", "c_hr": "fréquence",
        "c_rad": "Taux de radiation", "daily": "Check-ins quotidiens", "d_sleep": "Sommeil (h)", "d_mood": "Humeur (1-5)",
        "d_stress": "Stress (1-5)", "d_res": "Exercice (min)", "d_none": "Aucun check-in.",
        "why": "Pourquoi c'est important", "steps": "Mesures suggérées", "flags": "Indicateurs", "trend": "Tendance",
        "ok": "Paramètres nominaux.", "watch": "Valeurs proches des limites.",
        "disc": "⚠️ Suggestions automatiques; données simulées.",
        "tab1": "Moniteur", "tab2": "Comparaison d'équipage", "tab3": "Données NASA",
        "comp_h": "Comparaison de télémétrie", "comp_p": "Comparez tous les astronautes.",
        "live_h": "Données NASA en direct", "live_p": "Panneaux optionnels connectés aux API NASA.",
        "donki": "Événements solaires", "osdr": "Recherche OSDR",
        "term": "Terme de recherche", "none_ev": "Aucun événement.", "fail": "Service injoignable:",
        "src": "Sources", "src_md": "- Télémétrie simulée & limites NASA."
    },
    "es": {
        "title": "Salud de Astronautas y Telemetría", "side": "Panel de control", "astro": "Astronauta",
        "time": "Tiempo (predeterminado: último)", "win": "Ventana (horas)",
        "replay": "Control deslizante para reproducir escenarios.",
        "lvl": ["NORMAL", "PRECAUCIÓN", "ADVERTENCIA", "CRÍTICO"],
        "ctx": {"REST": "descanso", "EXERCISE": "ejercicio", "SLEEP": "sueño", "RECOVERY": "recuperación"},
        "recov": "↘ mejorando", "live": "Estado en vivo", "domains": "Dominios de riesgo",
        "dom": {"environment": "Cabina", "radiation": "Radiación", "cardio": "Cardio",
                "behavioral": "Conductual", "bone": "Hueso / Músculo", "immune": "Inmune"},
        "no_data": "sin datos", "immune_note": "requiere datos de laboratorio",
        "m_pco2": "pCO₂ (mmHg)", "m_p": "Presión (kPa)", "m_o2": "pO₂ (mmHg)", "m_hr": "Pulso (bpm)",
        "m_spo2": "SpO₂ (%)", "m_dose": "Dosis acumulada (mSv)", "m_dose_pct": "del límite",
        "c_pco2": "pCO₂ cabina", "c_p": "Presión cabina", "c_o2": "pO₂ cabina", "c_hr": "pulso",
        "c_rad": "Tasa de radiación", "daily": "Registros diarios", "d_sleep": "Sueño (h)", "d_mood": "Ánimo (1-5)",
        "d_stress": "Estrés (1-5)", "d_res": "Ejercicio (min)", "d_none": "Sin registros.",
        "why": "Por qué importa", "steps": "Pasos sugeridos", "flags": "Alertas activas", "trend": "Nota de tendencia",
        "ok": "Parámetros nominales.", "watch": "Valores cerca del límite.",
        "disc": "⚠️ Sugerencias de prototipo; datos simulados.",
        "tab1": "Monitor", "tab2": "Comparación de tripulación", "tab3": "Datos de NASA",
        "comp_h": "Comparación de telemetría", "comp_p": "Compare la tripulación.",
        "live_h": "Datos reales de NASA", "live_p": "Paneles opcionales con APIs públicas de NASA.",
        "donki": "Eventos solares (DONKI)", "osdr": "Buscar en OSDR",
        "term": "Término de búsqueda", "none_ev": "Sin eventos.", "fail": "No se pudo conectar:",
        "src": "Fuentes", "src_md": "- Telemetría simulada y límites NASA."
    }
}

st.set_page_config(page_title="Astronaut Health Monitor", page_icon="🛰️", layout="wide")

@st.cache_data
def load_tele(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    for key, col in COLS.items():
        lo, hi = PLAUSIBLE[key]
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    for c in ("pred_flags", "pred_notes", "pred_action", "pred_action_tr", "pred_why", "pred_why_tr"):
        df[c] = df[c].fillna("")
    return df

@st.cache_data
def load_daily(path):
    if not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    for c in ("daily_flags", "daily_action", "daily_action_tr", "daily_why", "daily_why_tr"):
        df[c] = df[c].fillna("")
    return df

def fmt(v, digits=1):
    return "—" if pd.isna(v) else f"{v:.{digits}f}"

def delta(cur_v, prev_v, digits=2):
    x = cur_v - prev_v
    return None if pd.isna(x) else f"{x:+.{digits}f}"

def line_chart(df, col, title, unit, rules=(), height=220, x="timestamp:T", point=False):
    base = alt.Chart(df).mark_line(strokeWidth=2, color="#4fc3f7", point=point).encode(
        x=alt.X(x, title=None),
        y=alt.Y(f"{col}:Q", title=unit, scale=alt.Scale(zero=False)),
        tooltip=[x, alt.Tooltip(f"{col}:Q", format=".2f")],
    )
    layers = [base]
    if rules:
        rdf = pd.DataFrame(list(rules), columns=["v", "color", "label"])
        layers.append(alt.Chart(rdf).mark_rule(strokeDash=[6, 4]).encode(
            y="v:Q", color=alt.Color("color:N", scale=None), tooltip=["label:N", "v:Q"]))
    return alt.layer(*layers).properties(title=title, height=height)

def multi_line_chart(df, col, title, unit, height=280):
    return alt.Chart(df).mark_line(strokeWidth=2).encode(
        x=alt.X("timestamp:T", title=None),
        y=alt.Y(f"{col}:Q", title=unit, scale=alt.Scale(zero=False)),
        color=alt.Color("astronaut_id:N", title="Astronaut"),
        tooltip=["timestamp:T", "astronaut_id:N", alt.Tooltip(f"{col}:Q", format=".2f")]
    ).properties(title=title, height=height).interactive()

def tile(label, status_text, color):
    return (f"<div style='background:{color};padding:10px 12px;border-radius:10px;color:white;min-height:78px;'>"
            f"<div style='font-size:12px;opacity:.9;'>{label}</div>"
            f"<div style='font-size:17px;font-weight:700;'>{status_text}</div></div>")

# ---------------- Sidebar & Dil Seçimi ----------------
lang_map = {"English": "en", "Türkçe": "tr", "Deutsch": "de", "Français": "fr", "Español": "es"}
selected_lang_name = st.sidebar.selectbox("🌐 Dil / Language / Sprache", list(lang_map.keys()))
lang = lang_map[selected_lang_name]
T = I18N[lang]
suffix = "_tr" if lang == "tr" else ""

st.sidebar.title("🛰️ " + T["side"])

# --- Karanlık Mod Seçici ---
st.sidebar.markdown("---")
theme_mode = st.sidebar.selectbox("🎨 Arayüz Teması", ["Uzay Karanlığı (Dark)", "Aydınlık (Light)"])

if theme_mode == "Uzay Karanlığı (Dark)":
    st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
        }
        </style>
    """, unsafe_allow_html=True)

df = load_tele(DATA_T)
daily = load_daily(DATA_D)

astro = st.sidebar.selectbox(T["astro"], sorted(df["astronaut_id"].unique()))
d_all = df[df["astronaut_id"] == astro].sort_values("timestamp").reset_index(drop=True)
t_min, t_max = d_all["timestamp"].min().to_pydatetime(), d_all["timestamp"].max().to_pydatetime()
t_sel = st.sidebar.slider(T["time"], min_value=t_min, max_value=t_max, value=t_max,
                          step=timedelta(minutes=5), format="DD/MM HH:mm")
window_h = st.sidebar.select_slider(T["win"], options=[2, 4, 6, 12, 24, 72], value=6)
st.sidebar.caption(T["replay"])

d = d_all[d_all["timestamp"] <= pd.Timestamp(t_sel)]
win = d[d["timestamp"] >= d["timestamp"].max() - pd.Timedelta(hours=window_h)]
cur = d.iloc[-1]
prev = d.iloc[-2] if len(d) > 1 else cur

tab_mon, tab_comp, tab_live = st.tabs([T["tab1"], T["tab2"], T["tab3"]])

with tab_mon:
    st.title(T["title"])
    lvl = int(cur["pred_level"])
    recov_html = "<div style='font-size:14px;'>" + T["recov"] + "</div>" if cur["pred_recovering"] else ""
    st.markdown(
        f"""<div style="background:{LEVEL_COLOR[lvl]};padding:18px 24px;border-radius:12px;color:white;">
        <div style="font-size:14px;opacity:.9;">{astro} · {cur['timestamp']:%d.%m.%Y %H:%M} UTC · {T['ctx'].get(cur['pred_context'], cur['pred_context'])} · {T['live']}</div>
        <div style="font-size:34px;font-weight:700;">{T['lvl'][lvl]}</div>
        {recov_html}
        </div>""", unsafe_allow_html=True)

    # ---- Risk Alanı Kutucukları ----
    st.write("")
    st.caption(T["domains"])
    if len(daily):
        dd_astro = daily[(daily["astronaut_id"] == astro) & (daily["date"].dt.date <= pd.Timestamp(t_sel).date())]
    else:
        dd_astro = pd.DataFrame()
    last_daily = dd_astro.iloc[-1] if len(dd_astro) else None
    tiles = []
    for key in ("environment", "radiation", "cardio"):
        lv = int(cur[f"dom_{key}"])
        tiles.append((T["dom"][key], T["lvl"][lv].split(" ")[0], LEVEL_COLOR[lv]))
    for key, col in (("behavioral", "dom_behavioral"), ("bone", "dom_bone")):
        if last_daily is None:
            tiles.append((T["dom"][key], T["no_data"], NA_COLOR))
        else:
            lv = int(last_daily[col])
            tiles.append((T["dom"][key], T["lvl"][lv].split(" ")[0], LEVEL_COLOR[lv]))
    tiles.append((T["dom"]["immune"], T["no_data"], NA_COLOR))
    for c, (label, text, color) in zip(st.columns(6), tiles):
        c.markdown(tile(label, text, color), unsafe_allow_html=True)
    st.caption("ⓘ " + T["dom"]["immune"] + ": " + T["immune_note"])

    # ---- Metrikler ----
    st.write("")
    m = st.columns(6)
    m[0].metric(T["m_pco2"], fmt(cur["cabin_pco2_mmhg"], 2), delta(cur["cabin_pco2_mmhg"], prev["cabin_pco2_mmhg"]), delta_color="inverse")
    m[1].metric(T["m_p"], fmt(cur["cabin_pressure_kpa"], 2), delta(cur["cabin_pressure_kpa"], prev["cabin_pressure_kpa"]))
    m[2].metric(T["m_o2"], fmt(cur["cabin_po2_mmhg"], 1))
    m[3].metric(T["m_hr"], fmt(cur["heart_rate_bpm"], 0))
    m[4].metric(T["m_spo2"], fmt(cur["spo2_percent"], 1))
    m[5].metric(T["m_dose"], fmt(cur["cum_dose_msv"], 3), f"{cur['cum_dose_pct_career']:.4f}% " + T["m_dose_pct"], delta_color="off")

    # ---- Grafikler ----
    RED, AMB, BLU = "#c62828", "#f9a825", "#1976d2"
    pc, pr, po, ra = (LIMITS_REST[k] for k in ("pco2", "pressure", "po2", "radiation"))
    c1, c2 = st.columns(2)
    c1.altair_chart(line_chart(win, "cabin_pco2_mmhg", T["c_pco2"], "mmHg",
                               [(pc.high[0], BLU, "caution"), (pc.high[1], AMB, "warning"), (pc.high[2], RED, "critical")]),
                        use_container_width=True)
    c2.altair_chart(line_chart(win, "cabin_pressure_kpa", T["c_p"], "kPa",
                               [(pr.low[2], BLU, "caution"), (pr.low[1], AMB, "warning"), (pr.low[0], RED, "critical")]),
                        use_container_width=True)
    c3, c4, c5 = st.columns(3)
    c3.altair_chart(line_chart(win, "cabin_po2_mmhg", T["c_o2"], "mmHg",
                               [(po.low[2], BLU, "caution"), (po.low[1], AMB, "warning"), (po.low[0], RED, "critical")]),
                        use_container_width=True)
    c4.altair_chart(line_chart(win, "radiation_rate_usv_day", T["c_rad"], "µSv/day",
                               [(ra.high[0], BLU, "caution"), (ra.high[1], AMB, "warning"), (ra.high[2], RED, "critical")]),
                        use_container_width=True)
    c5.altair_chart(line_chart(win, "heart_rate_bpm", f"{astro} {T['c_hr']}", "bpm"), use_container_width=True)

    # ---- Günlük Check-in'ler ----
    st.subheader(T["daily"])
    if len(dd_astro):
        specs = [("sleep_hours", T["d_sleep"], "h"), ("mood_1_5", T["d_mood"], ""),
                 ("stress_1_5", T["d_stress"], ""), ("resistive_exercise_min", T["d_res"], "min")]
        for col_, (c_name, title, unit) in zip(st.columns(4), specs):
            col_.altair_chart(line_chart(dd_astro, c_name, title, unit, height=170, x="date:T", point=True),
                              use_container_width=True)
    else:
        st.info(T["d_none"])

    # ---- Önerilen Adımlar ----
    st.subheader(T["steps"])
    flags = [f for f in cur["pred_flags"].split(" | ") if f]
    why = cur["pred_why" + suffix] if lang == "tr" else cur["pred_why"]
    action = cur["pred_action" + suffix] if lang == "tr" else cur["pred_action"]
    daily_on = last_daily is not None and int(last_daily["daily_level"]) >= 1
    if daily_on:
        why = (why + " " + last_daily.get("daily_why" + suffix, "")).strip()
        action = " ; ".join(x for x in (action, last_daily.get("daily_action" + suffix, "")) if x)
    if lvl == 0 and not flags and not daily_on:
        st.success(T["ok"])
    else:
        body = []
        if action:
            body.append("**" + T["steps"] + ":** " + action)
        if why:
            body.append("**" + T["why"] + ":** " + why)
        if cur["pred_notes"]:
            body.append("**" + T["trend"] + ":** " + cur["pred_notes"])
        if flags:
            body.append("**" + T["flags"] + ":**\n" + "\n".join(f"- `{f}`" for f in flags))
        box = st.error if lvl == 3 else st.warning
        box("\n\n".join(body) if body else T["watch"])
    st.caption(T["disc"])
    with st.expander(T["src"]):
        st.markdown(T["src_md"])

with tab_comp:
    st.subheader(T["comp_h"])
    st.write(T["comp_p"])
    
    # Tüm astronotların seçilen zaman penceresindeki verilerini filtrele
    df_win = df[df["timestamp"] <= pd.Timestamp(t_sel)]
    df_win = df_win[df_win["timestamp"] >= df_win["timestamp"].max() - pd.Timedelta(hours=window_h)]
    
    col_comp1, col_comp2 = st.columns(2)
    with col_comp1:
        st.altair_chart(multi_line_chart(df_win, "heart_rate_bpm", "Heart Rate Comparison (bpm)", "bpm"), use_container_width=True)
    with col_comp2:
        st.altair_chart(multi_line_chart(df_win, "cum_dose_msv", "Cumulative Radiation Dose (mSv)", "mSv"), use_container_width=True)

with tab_live:
    st.subheader(T["live_h"])
    st.write(T["live_p"])
    if st.button(T["donki"]):
        try:
            ev = real_data.fetch_donki_sep(date.today() - timedelta(days=59), date.today())
            if len(ev):
                st.dataframe(ev, use_container_width=True)
            else:
                st.info(T["none_ev"])
        except Exception as e:  # noqa: BLE001
            st.warning(f"{T['fail']} {e}")
    term = st.text_input(T["term"], "radiation")
    if st.button(T["osdr"]):
        try:
            st.dataframe(real_data.fetch_osdr_studies(term), use_container_width=True)
        except Exception as e:  # noqa: BLE001
            st.warning(f"{T['fail']} {e}")
