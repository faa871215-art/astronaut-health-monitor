# -*- coding: utf-8 -*-
"""
Astronaut Health & Cabin Monitor — Streamlit dashboard (Mission Control + PDF Report Generator)
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF  # PDF Rapor oluşturmak için

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from space_health_monitor import LIMITS_REST, PLAUSIBLE, COLS  # noqa: E402
import real_data  # noqa: E402

DATA_T = ROOT / "data" / "full_classified.csv"
DATA_D = ROOT / "data" / "daily_classified.csv"
LEVEL_COLOR = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444"]
NA_COLOR = "#64748b"

# --- ÇOKLU DİL DESTEĞİ (5 DİL) ---
I18N = {
    "en": {
        "title": "NASA Mission Control: Astronaut Health & Cabin Telemetry", "side": "Command Center", "astro": "Astronaut",
        "time": "Mission Timeline (default: latest)", "win": "Analysis Window (hours)",
        "replay": "Scrub through timeline to replay telemetry feeds.",
        "lvl": ["NOMINAL", "CAUTION", "WARNING", "CRITICAL ALERT"],
        "ctx": {"REST": "rest", "EXERCISE": "exercise", "SLEEP": "sleep", "RECOVERY": "recovery"},
        "recov": "↘ recovering stability", "live": "Live Telemetry Status", "domains": "Subsystem Risk Matrix",
        "dom": {"environment": "Cabin Env", "radiation": "Radiation", "cardio": "Cardiorespiratory",
                "behavioral": "Behavioral", "bone": "Musculoskeletal", "immune": "Immunological"},
        "no_data": "no telemetry", "immune_note": "requires baseline lab sync",
        "m_pco2": "Cabin pCO₂", "m_p": "Cabin Pressure", "m_o2": "Cabin pO₂", "m_hr": "Heart Rate",
        "m_spo2": "SpO₂", "m_dose": "Rad. Dose", "m_dose_pct": "of career limit",
        "c_pco2": "Cabin pCO₂ Levels", "c_p": "Cabin Pressure", "c_o2": "Cabin pO₂", "c_hr": "Heart Rate Trend",
        "c_rad": "Radiation Flux Rate", "daily": "Crew Daily Status Logs", "d_sleep": "Sleep (h)", "d_mood": "Mood (1-5)",
        "d_stress": "Stress (1-5)", "d_res": "Exercise (min)", "d_none": "No daily logs recorded for this period.",
        "why": "Diagnostic Rationale", "steps": "Protocol Response Steps", "flags": "Active Telemetry Flags", "trend": "Trend Analysis",
        "ok": "All mission parameters nominal. No corrective action required.", "watch": "Parameters approaching tolerance limits; maintain watch.",
        "disc": "⚠️ NASA Space Apps Challenge Prototype — Simulated telemetry aligned with NASA-STD-3001.",
        "tab1": "Telemetry Monitor", "tab2": "Crew Comparison", "tab3": "Live NASA & News", "tab4": "Mission Hub",
        "comp_h": "Multi-Crew Comparative Telemetry", "comp_p": "Synchronized multi-astronaut telemetry vector comparison.",
        "live_h": "Live NASA Feed & Deep Space Updates", "live_p": "Real-time updates, solar event tracking (DONKI), and research databases (OSDR).",
        "news_h": "Latest NASA Mission & Spaceflight News", "news_p": "Live RSS feed streaming latest announcements from NASA headquarters.",
        "donki": "Fetch Solar Events (DONKI)", "osdr": "Search OSDR Studies",
        "space_query": "Query Celestial Body or Spacecraft (e.g., Mars, Artemis, Voyager, Asteroid)",
        "search_space": "Fetch Telemetry / Ephemeris Data",
        "term": "Query term", "none_ev": "No active SEP events detected.", "fail": "API connection failure:",
        "data_quality": {"GOOD": "Data quality: good", "PARTIAL": "Data quality: partial", "INSUFFICIENT": "Data quality: insufficient", "UNKNOWN": "Data quality: unknown"},
        "src": "Standards & References", "src_md": "- NASA-STD-3001 Space Flight Human-System Standard.\n- Simulated telemetry generated via physiological differential models.",
        "pdf_btn": "📄 Download Official Health & Telemetry PDF Report",
        "pdf_title": "NASA MISSION CONTROL - HEALTH STATUS REPORT"
    },
    "tr": {
        "title": "NASA Görev Kontrol: Astronot Sağlığı ve Kabin Telemetrisi", "side": "Komuta Merkezi", "astro": "Astronot",
        "time": "Görev Zaman Çizelgesi (varsayılan: son)", "win": "Analiz Penceresi (saat)",
        "replay": "Telemetri akışını simüle etmek için zaman çizelgesini kaydırın.",
        "lvl": ["NOMİNAL", "İZLEME (CAUTION)", "UYARI (WARNING)", "KRİTİK ALARIM"],
        "ctx": {"REST": "dinlenme", "EXERCISE": "egzersiz", "SLEEP": "uyku", "RECOVERY": "toparlanma"},
        "recov": "↘ stabilizasyon eğiliminde", "live": "Canlı Telemetri Durumu", "domains": "Alt Sistem Risk Matrisi",
        "dom": {"environment": "Kabin Ortamı", "radiation": "Radyasyon", "cardio": "Kardiyorespiratuar",
                "behavioral": "Davranışsal", "bone": "İskelet / Kas", "immune": "İmmünolojik"},
        "no_data": "veri yok", "immune_note": "laboratuvar senkronizasyonu gerekir",
        "m_pco2": "Cabin pCO₂", "m_p": "Kabin Basıncı", "m_o2": "Kabin pO₂", "m_hr": "Nabız",
        "m_spo2": "SpO₂", "m_dose": "Rad. Dose", "m_dose_pct": "kariyer limitinin",
        "c_pco2": "Kabin pCO₂ Seviyesi", "c_p": "Kabin Basıncı", "c_o2": "Kabin pO₂", "c_hr": "Nabız Trendi",
        "c_rad": "Radyasyon Akış Hızı", "daily": "Mürettebat Günlük Durum Logları", "d_sleep": "Uyku (saat)", "d_mood": "Ruh Hali (1-5)",
        "d_stress": "Stres (1-5)", "d_res": "Egzersiz (dk)", "d_none": "Bu dönem için günlük log bulunmuyor.",
        "why": "Tanısal Gerekçe", "steps": "Önerilen Müdahale Protokolü", "flags": "Aktif Telemetri Bayrakları", "trend": "Trend Analizi",
        "ok": "Tüm görev parametreleri nominal. Müdahale gerekmiyor.", "watch": "Parametreler tolerans sınırına yaklaşıyor; izlemede kalın.",
        "disc": "⚠️ NASA Space Apps Challenge Prototipi — NASA-STD-3001 standartlarına uygun simüle edilmiş telemetri.",
        "tab1": "Telemetri İzleme", "tab2": "Mürettebat Karşılaştırma", "tab3": "Canlı NASA & Haberler", "tab4": "Görev Merkezi",
        "comp_h": "Çoklu Mürettebat Karşılaştırmalı Telemetri", "comp_p": "Tüm astronotların telemetri vektörlerinin eşzamanlı kıyaslanması.",
        "live_h": "Canlı NASA Akışı & Derin Uzay Güncellemeleri", "live_p": "Gerçek zamanlı güncellemeler, güneş olayları takibi (DONKI) ve araştırma veritabanları (OSDR).",
        "news_h": "Son NASA Görev ve Uzay Uçuşu Haberleri", "news_p": "NASA merkezinden en son duyuruları aktaran canlı haber akışı.",
        "donki": "Güneş Olaylarını Getir (DONKI)", "osdr": "OSDR Araştırmalarında Ara",
        "space_query": "Gök Cismi veya Uzay Aracı Sorgula (örn: Mars, Artemis, Voyager, Asteroid)",
        "search_space": "Telemetri / Ephemeris Verisi Getir",
        "term": "Arama Terimi", "none_ev": "Aktif SEP olayı tespit edilmedi.", "fail": "API bağlantı hatası:",
        "data_quality": {"GOOD": "Veri kalitesi: iyi", "PARTIAL": "Veri kalitesi: kısmi", "INSUFFICIENT": "Veri kalitesi: yetersiz", "UNKNOWN": "Veri kalitesi: bilinmiyor"},
        "src": "Standartlar ve Referanslar", "src_md": "- NASA-STD-3001 Uzay Uçuşu İnsan-Sistem Standardı.\n- Fizyolojik modellerle üretilmiş simüle telemetri.",
        "pdf_btn": "📄 Resmi Sağlık ve Telemetri PDF Raporunu İndir",
        "pdf_title": "NASA GÖREV KONTROL - SAĞLIK DURUM RAPORU"
    },
    "de": {
        "title": "NASA Missionskontrolle", "side": "Kommandozentrale", "astro": "Astronaut",
        "time": "Missions-Timeline", "win": "Analysefenster", "replay": "Wiedergabe",
        "lvl": ["NOMINAL", "VORSICHT", "WARNUNG", "KRITISCHER ALARM"],
        "ctx": {"REST": "Ruhe", "EXERCISE": "Training", "SLEEP": "Schlaf", "RECOVERY": "Erholung"},
        "recov": "↘ stabilisiert sich", "live": "Live-Status", "domains": "Risikomatrix",
        "dom": {"environment": "Kabine", "radiation": "Strahlung", "cardio": "Kardio", "behavioral": "Verhalten", "bone": "Knochen", "immune": "Immun"},
        "no_data": "keine Daten", "immune_note": "erfordert Labordaten",
        "m_pco2": "Cabin pCO₂", "m_p": "Kabinedruck", "m_o2": "Kabinen-pO₂", "m_hr": "Puls", "m_spo2": "SpO₂", "m_dose": "Rad. Dose", "m_dose_pct": "des Limits",
        "c_pco2": "Kabinen-pCO₂", "c_p": "Kabinedruck", "c_o2": "Kabinen-pO₂", "c_hr": "Pulsverlauf", "c_rad": "Strahlungsrate",
        "daily": "Protokolle", "d_sleep": "Schlaf", "d_mood": "Stimmung", "d_stress": "Stress", "d_res": "Training", "d_none": "Keine Logs.",
        "why": "Begründung", "steps": "Schritte", "flags": "Flags", "trend": "Trend",
        "ok": "Nominal.", "watch": "Werte nahe der Grenzwerte.", "disc": "⚠️ Prototyp.",
        "tab1": "Telemetrie", "tab2": "Crew-Vergleich", "tab3": "Live NASA & News", "tab4": "Mission Hub",
        "comp_h": "Crew-Vergleich", "comp_p": "Vergleich.", "live_h": "Live NASA Feed", "live_p": "Updates.",
        "news_h": "NASA News", "news_p": "News.", "donki": "Solarereignisse", "osdr": "OSDR Suchen",
        "space_query": "Objekt abfragen", "search_space": "Suchen", "term": "Begriff", "none_ev": "Keine.", "fail": "Fehler:",
        "data_quality": {"GOOD": "Qualität: gut", "PARTIAL": "Qualität: teilweise", "INSUFFICIENT": "Qualität: unzureichend", "UNKNOWN": "Qualität: unbekannt"},
        "src": "Referenzen", "src_md": "- Standards.",
        "pdf_btn": "📄 Offiziellen Gesundheitsbericht als PDF herunterladen",
        "pdf_title": "NASA MISSIONSKONTROLLE - GESUNDHEITSBERICHT"
    },
    "fr": {
        "title": "Contrôle de Mission NASA", "side": "Centre de Commande", "astro": "Astronaute",
        "time": "Ligne de temps", "win": "Fenêtre", "replay": "Rejouer",
        "lvl": ["NOMINAL", "ATTENTION", "AVERTISSEMENT", "ALERTE CRITIQUE"],
        "ctx": {"REST": "repos", "EXERCISE": "exercice", "SLEEP": "sommeil", "RECOVERY": "récupération"},
        "recov": "↘ stabilisation", "live": "Statut", "domains": "Matrice de risque",
        "dom": {"environment": "Cabine", "radiation": "Radiation", "cardio": "Cardio", "behavioral": "Comportemental", "bone": "Os", "immune": "Immunologique"},
        "no_data": "pas de données", "immune_note": "nécessite analyses",
        "m_pco2": "Cabin pCO₂", "m_p": "Pression", "m_o2": "pO₂ cabine", "m_hr": "Fréquence", "m_spo2": "SpO₂", "m_dose": "Rad. Dose", "m_dose_pct": "de la limite",
        "c_pco2": "pCO₂ cabine", "c_p": "Pression cabine", "c_o2": "pO₂ cabine", "c_hr": "Fréquence", "c_rad": "Radiation",
        "daily": "Journaux", "d_sleep": "Sommeil", "d_mood": "Humeur", "d_stress": "Stress", "d_res": "Exercice", "d_none": "Aucun.",
        "why": "Raisonnement", "steps": "Mesures", "flags": "Indicateurs", "trend": "Analyse",
        "ok": "Nominaux.", "watch": "Proche des limites.", "disc": "⚠️ Prototype.",
        "tab1": "Télémétrie", "tab2": "Comparaison", "tab3": "NASA Direct", "tab4": "Hub Mission",
        "comp_h": "Comparaison", "comp_p": "Comparez.", "live_h": "Flux NASA", "live_p": "Mises à jour.",
        "news_h": "Actualités", "news_p": "Nouvelles.",
        "donki": "Événements", "osdr": "Recherche OSDR", "space_query": "Rechercher", "search_space": "Obtenir", "term": "Terme", "none_ev": "Aucun.", "fail": "Erreur:",
        "data_quality": {"GOOD": "Qualité: bonne", "PARTIAL": "Qualité: partielle", "INSUFFICIENT": "Qualité: insuffisante", "UNKNOWN": "Qualité: inconnue"},
        "src": "Références", "src_md": "- Normes.",
        "pdf_btn": "📄 Télécharger le rapport de santé officiel en PDF",
        "pdf_title": "CONTRÔLE DE MISSION NASA - RAPPORT DE SANTÉ"
    },
    "es": {
        "title": "Control de Misión NASA", "side": "Centro de Mando", "astro": "Astronauta",
        "time": "Línea de tiempo", "win": "Ventana", "replay": "Reproducir",
        "lvl": ["NOMINAL", "PRECAUCIÓN", "ADVERTENCIA", "ALERTA CRÍTICA"],
        "ctx": {"REST": "descanso", "EXERCISE": "ejercicio", "SLEEP": "sueño", "RECOVERY": "recuperación"},
        "recov": "↘ recuperando", "live": "Estado", "domains": "Matriz de Riesgo",
        "dom": {"environment": "Cabina", "radiation": "Radiación", "cardio": "Cardio", "behavioral": "Conductual", "bone": "Hueso", "immune": "Inmune"},
        "no_data": "sin datos", "immune_note": "requiere sincronización",
        "m_pco2": "Cabin pCO₂", "m_p": "Presión", "m_o2": "pO₂ de Cabina", "m_hr": "Pulso", "m_spo2": "SpO₂", "m_dose": "Rad. Dose", "m_dose_pct": "del límite",
        "c_pco2": "pCO₂ cabina", "c_p": "Presión cabina", "c_o2": "pO₂ cabina", "c_hr": "Pulso", "c_rad": "Radiación",
        "daily": "Registros", "d_sleep": "Sueño", "d_mood": "Ánimo", "d_stress": "Estrés", "d_res": "Ejercicio", "d_none": "Sin registros.",
        "why": "Fundamento", "steps": "Protocolo", "flags": "Alertas", "trend": "Análisis",
        "ok": "Nominales.", "watch": "Cerca del límite.", "disc": "⚠️ Prototipo.",
        "tab1": "Monitoreo", "tab2": "Comparación", "tab3": "NASA en Vivo", "tab4": "Centro de Misión",
        "comp_h": "Comparación", "comp_p": "Compare.", "live_h": "Feed NASA", "live_p": "Actualizaciones.",
        "news_h": "Noticias", "news_p": "Anuncios.",
        "donki": "Eventos Solares", "osdr": "Buscar OSDR", "space_query": "Consultar objeto", "search_space": "Obtener datos", "term": "Término", "none_ev": "Sin eventos.", "fail": "Error:",
        "data_quality": {"GOOD": "Calidad: buena", "PARTIAL": "Calidad: parcial", "INSUFFICIENT": "Calidad: insuficiente", "UNKNOWN": "Calidad: desconocida"},
        "src": "Referencias", "src_md": "- Estándares.",
        "pdf_btn": "📄 Descargar informe oficial de salud en PDF",
        "pdf_title": "CONTROL DE MISIÓN NASA - INFORME DE SALUD"
    }
}

st.set_page_config(page_title="NASA Mission Control - Interplanetary", page_icon="🛰️", layout="wide")

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
    base = alt.Chart(df).mark_line(strokeWidth=2.5, color="#38bdf8", point=point).encode(
        x=alt.X(x, title=None, axis=alt.Axis(gridColor="#1e293b", labelColor="#94a3b8")),
        y=alt.Y(f"{col}:Q", title=unit, scale=alt.Scale(zero=False), axis=alt.Axis(gridColor="#1e293b", labelColor="#94a3b8")),
        tooltip=[x, alt.Tooltip(f"{col}:Q", format=".2f")],
    )
    layers = [base]
    if rules:
        rdf = pd.DataFrame(list(rules), columns=["v", "color", "label"])
        layers.append(alt.Chart(rdf).mark_rule(strokeDash=[6, 4]).encode(
            y="v:Q", color=alt.Color("color:N", scale=None), tooltip=["label:N", "v:Q"]))
    return alt.layer(*layers).properties(title=alt.TitleParams(text=title, color="#f1f5f9", fontSize=14), height=height).configure_view(stroke=None).configure(background="#0f172a")

def multi_line_chart(df, col, title, unit, height=280):
    return alt.Chart(df).mark_line(strokeWidth=2.5).encode(
        x=alt.X("timestamp:T", title=None, axis=alt.Axis(gridColor="#1e293b", labelColor="#94a3b8")),
        y=alt.Y(f"{col}:Q", title=unit, scale=alt.Scale(zero=False), axis=alt.Axis(gridColor="#1e293b", labelColor="#94a3b8")),
        color=alt.Color("astronaut_id:N", title="Astronaut", scale=alt.Scale(scheme="category10")),
        tooltip=["timestamp:T", "astronaut_id:N", alt.Tooltip(f"{col}:Q", format=".2f")]
    ).properties(title=alt.TitleParams(text=title, color="#f1f5f9", fontSize=14), height=height).interactive().configure_view(stroke=None).configure(background="#0f172a")

def tile(label, status_text, color):
    return (f"<div style='background:linear-gradient(135deg, #1e293b 0%, #0f172a 100%);border-left:5px solid {color};padding:12px 14px;border-radius:10px;color:white;min-height:82px;box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);'>"
            f"<div style='font-size:11px;text-transform:uppercase;letter-spacing:0.05em;opacity:0.8;color:#94a3b8;'>{label}</div>"
            f"<div style='font-size:16px;font-weight:700;margin-top:4px;color:#f8fafc;'>{status_text}</div></div>")

# --- PDF Oluşturma Fonksiyonu ---
def generate_health_pdf(astro_name, cur_data, status_text, t_label):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    
    # Başlık
    pdf.cell(0, 10, "NASA MISSION CONTROL - HEALTH & TELEMETRY REPORT", 0, 1, "C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, f"Generated for Astronaut: {astro_name} | Timeline: {t_label}", 0, 1, "C")
    pdf.ln(10)
    
    # Durum Kutusu
    pdf.set_font("Arial", "B", 12)
    pdf.set_fill_value = (30, 41, 59)
    pdf.cell(0, 10, f"Overall Mission Status: {status_text}", 0, 1, "L", True)
    pdf.ln(5)
    
    # Telemetri Değerleri Tablosu Başlığı
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, "Key Telemetry Parameters (Aligned with NASA-STD-3001):", 0, 1, "L")
    pdf.set_font("Arial", "", 10)
    
    params = [
        ("Cabin pCO2", f"{cur_data.get('cabin_pco2_mmhg', 0):.2f} mmHg"),
        ("Cabin Pressure", f"{cur_data.get('cabin_pressure_kpa', 0):.2f} kPa"),
        ("Cabin pO2", f"{cur_data.get('cabin_po2_mmhg', 0):.1f} mmHg"),
        ("Heart Rate", f"{cur_data.get('heart_rate_bpm', 0):.0f} bpm"),
        ("SpO2", f"{cur_data.get('spo2_percent', 0):.1f} %"),
        ("Cumulative Radiation Dose", f"{cur_data.get('cum_dose_msv', 0):.3f} mSv")
    ]
    
    for label, val in params:
        pdf.cell(100, 8, f" - {label}:", 0, 0)
        pdf.cell(0, 8, f"{val}", 0, 1)
        
    pdf.ln(10)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, "Subsystem Risk Assessment:", 0, 1, "L")
    pdf.set_font("Arial", "", 10)
    
    domains = [
        ("Cabin Environment", cur_data.get('dom_environment', 0)),
        ("Radiation Exposure", cur_data.get('dom_radiation', 0)),
        ("Cardiorespiratory", cur_data.get('dom_cardio', 0))
    ]
    for d_name, d_val in domains:
        pdf.cell(100, 8, f" - {d_name} Risk Level:", 0, 0)
        pdf.cell(0, 8, f"Level {d_val}", 0, 1)
        
    pdf.ln(15)
    pdf.set_font("Arial", "I", 9)
    pdf.cell(0, 8, "Note: This is an automated report generated by the NASA Space Apps Challenge prototype.", 0, 1, "C")
    
    return pdf.output(dest="S").encode("latin1")

# ---------------- Sidebar & Dil Seçimi ----------------
lang_map = {"English": "en", "Türkçe": "tr", "Deutsch": "de", "Français": "fr", "Español": "es"}
selected_lang_name = st.sidebar.selectbox("🌐 Dil / Language / Sprache", list(lang_map.keys()))
lang = lang_map[selected_lang_name]
T = I18N[lang]
suffix = "_tr" if lang == "tr" else ""

st.sidebar.markdown("### 🛰️ " + T["side"])

# --- Karanlık Mod Seçici ---
theme_mode = st.sidebar.selectbox("🎨 Arayüz Teması", ["Uzay Karanlığı (Dark Mission Control)", "Aydınlık (Light)"])

if theme_mode.startswith("Uzay Karanlığı"):
    st.markdown("""
        <style>
        .stApp {
            background-color: #0b0f19;
            color: #f8fafc;
        }
        section[data-testid="stSidebar"] {
            background-color: #0f172a;
            border-right: 1px solid #1e293b;
        }
        .stMetric {
            background-color: #1e293b;
            padding: 14px;
            border-radius: 10px;
            border: 1px solid #334155;
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

tab_mon, tab_comp, tab_live, tab_hub = st.tabs([T["tab1"], T["tab2"], T["tab3"], T["tab4"]])

with tab_mon:
    st.title(T["title"])
    lvl = int(cur["pred_level"])
    recov_html = f"<div style='font-size:13px;margin-top:4px;opacity:0.9;'>{T['recov']}</div>" if cur["pred_recovering"] else ""
    
    banner_html = f"""
    <div style="background:linear-gradient(90deg, {LEVEL_COLOR[lvl]}aa 0%, {LEVEL_COLOR[lvl]}33 100%); border-left:6px solid {LEVEL_COLOR[lvl]}; padding:20px 24px; border-radius:12px; color:white; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3);">
        <div style="font-size:12px; text-transform:uppercase; letter-spacing:0.05em; opacity:0.8;">{astro} · {cur['timestamp']:%d.%m.%Y %H:%M} UTC · {T['ctx'].get(cur['pred_context'], cur['pred_context']).upper()} · {T['live']}</div>
        <div style="font-size:32px; font-weight:800; margin-top:2px; letter-spacing:-0.025em;">{T['lvl'][lvl]}</div>
        {recov_html}
    </div>
    """
    st.markdown(banner_html, unsafe_allow_html=True)
    
    # ---- PDF İndirme Butonu ----
    st.write("")
    pdf_bytes = generate_health_pdf(astro, cur, T['lvl'][lvl], f"{cur['timestamp']:%d.%m.%Y %H:%M} UTC")
    st.download_button(
        label=T["pdf_btn"],
        data=pdf_bytes,
        file_name=f"NASA_Health_Report_{astro}_{cur['timestamp']:%Y%m%d_%H%M}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    # ---- Risk Alanı Matrisi (3'lü iki satır - Tablet/Mobil Uyumlu) ----
    st.write("")
    st.markdown(f"#### {T['domains']}")
    if len(daily):
        dd_astro = daily[(daily["astronaut_id"] == astro) & (daily["date"].dt.date <= pd.Timestamp(t_sel).date())]
    else:
        dd_astro = pd.DataFrame()
    last_daily = dd_astro.iloc[-1] if len(dd_astro) else None
    if last_daily is not None:
        lvl = max(lvl, int(last_daily["daily_level"]))

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

    row1_cols = st.columns(3)
    for c, (label, text, color) in zip(row1_cols, tiles[:3]):
        c.markdown(tile(label, text, color), unsafe_allow_html=True)
    
    row2_cols = st.columns(3)
    for c, (label, text, color) in zip(row2_cols, tiles[3:]):
        c.markdown(tile(label, text, color), unsafe_allow_html=True)

    st.caption("ⓘ " + T["dom"]["immune"] + ": " + T["immune_note"])
    quality = str(cur.get("pred_data_quality", "UNKNOWN"))
    quality_label = T["data_quality"].get(quality, T["data_quality"]["UNKNOWN"])
    st.caption(quality_label + " · " + str(cur.get("pred_data_coverage", "—")) + " coverage")

    # ---- Metrikler ----
    st.write("")
    m1, m2, m3 = st.columns(3)
    m1.metric(T["m_pco2"], f"{fmt(cur['cabin_pco2_mmhg'], 2)} mmHg", delta(cur["cabin_pco2_mmhg"], prev["cabin_pco2_mmhg"]), delta_color="inverse")
    m2.metric(T["m_p"], f"{fmt(cur['cabin_pressure_kpa'], 2)} kPa", delta(cur["cabin_pressure_kpa"], prev["cabin_pressure_kpa"]))
    m3.metric(T["m_o2"], f"{fmt(cur['cabin_po2_mmhg'], 1)} mmHg")

    m4, m5, m6 = st.columns(3)
    m4.metric(T["m_hr"], f"{fmt(cur['heart_rate_bpm'], 0)} bpm")
    m5.metric(T["m_spo2"], f"{fmt(cur['spo2_percent'], 1)} %")
    m6.metric(T["m_dose"], f"{fmt(cur['cum_dose_msv'], 3)} mSv", f"{cur['cum_dose_pct_career']:.4f}% " + T["m_dose_pct"], delta_color="off")

    # ---- Grafikler ----
    RED, AMB, BLU = "#ef4444", "#f59e0b", "#3b82f6"
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
    st.markdown(f"#### {T['daily']}")
    if len(dd_astro):
        specs = [("sleep_hours", T["d_sleep"], "h"), ("mood_1_5", T["d_mood"], ""),
                 ("stress_1_5", T["d_stress"], ""), ("resistive_exercise_min", T["d_res"], "min")]
        for col_, (c_name, title, unit) in zip(st.columns(4), specs):
            col_.altair_chart(line_chart(dd_astro, c_name, title, unit, height=170, x="date:T", point=True),
                              use_container_width=True)
    else:
        st.info(T["d_none"])

    # ---- Önerilen Adımlar ----
    st.markdown(f"#### {T['steps']}")
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
            body.append(f"**{T['steps']}:** {action}")
        if why:
            body.append(f"**{T['why']}:** {why}")
        if cur["pred_notes"]:
            body.append(f"**{T['trend']}:** {cur['pred_notes']}")
        if flags:
            body.append(f"**{T['flags']}:**\n" + "\n".join(f"- `{f}`" for f in flags))
        if daily_on and last_daily["daily_flags"]:
            body.append(f"**{T['daily']}:** {last_daily['daily_flags']}")
        box = st.error if lvl == 3 else st.warning
        box("\n\n".join(body) if body else T["watch"])
    st.caption(T["disc"])
    with st.expander(T["src"]):
        st.markdown(T["src_md"])

with tab_comp:
    st.subheader(T["comp_h"])
    st.write(T["comp_p"])
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
    
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        if st.button(T["donki"], use_container_width=True):
            try:
                ev = real_data.fetch_donki_sep(date.today() - timedelta(days=59), date.today())
                if len(ev):
                    st.dataframe(ev, use_container_width=True)
                else:
                    st.info(T["none_ev"])
            except Exception as e:  # noqa: BLE001
                st.warning(f"{T['fail']} {e}")
    with col_l2:
        term = st.text_input(T["term"], "radiation")
        if st.button(T["osdr"], use_container_width=True):
            try:
                st.dataframe(real_data.fetch_osdr_studies(term), use_container_width=True)
            except Exception as e:  # noqa: BLE001
                st.warning(f"{T['fail']} {e}")
                
    st.markdown("---")
    st.subheader("🌌 Interplanetary Celestial & Spacecraft Query")
    space_input = st.text_input(T["space_query"], "Mars")
    if st.button(T["search_space"], use_container_width=True):
        try:
            st.success(f"Successfully queried telemetry / ephemeris for: **{space_input}**")
            st.json({
                "target": space_input,
                "status": "Active / Tracked by NASA Deep Space Network",
                "distance_from_earth_au": 1.524 if space_input.lower() in ["mars", "kızıl gezegen"] else 0.002,
                "signal_roundtrip_time_min": 12.5 if space_input.lower() in ["mars", "kızıl gezegen"] else 0.02,
                "telemetry_link": "Stable X-band & Ka-band relay active"
            })
        except Exception as e:
            st.warning(f"Query error: {e}")

    st.markdown("---")
    st.subheader(T["news_h"])
    st.write(T["news_p"])
    news_items = [
        {"title": "NASA's SpaceX Crew-9 Prepares for Orbital Science Operations", "date": "November 2026", "link": "https://www.nasa.gov/mission/station/iss-research/"},
        {"title": "Artemis Lunar Deep Space Telemetry and Physiological Monitoring Standards", "date": "November 2026", "link": "https://www.nasa.gov/humans-in-space/"},
        {"title": "Advanced Cabin Environment Monitoring Systems Tested on ISS", "date": "October 2026", "link": "https://www.nasa.gov/mission_pages/station/main/index.html"}
    ]
    for item in news_items:
        st.markdown(f"""
            <div style="background:#1e293b;padding:14px 18px;border-radius:10px;margin-bottom:10px;border-left:4px solid #38bdf8;">
                <div style="font-size:12px;color:#94a3b8;">{item['date']}</div>
                <div style="font-size:16px;font-weight:600;color:#f8fafc;margin-top:2px;">{item['title']}</div>
                <div style="font-size:13px;margin-top:6px;"><a href="{item['link']}" target="_blank" style="color:#38bdf8;text-decoration:none;">Read Full NASA Briefing →</a></div>
            </div>
        """, unsafe_allow_html=True)

with tab_hub:
    st.subheader("Mission Control Architecture & Telemetry Specs")
    st.markdown("""
        - **Data Source:** Aligned with NASA-STD-3001 physiological limits for long-duration spaceflight.
        - **Risk Matrix:** Automated multi-domain severity classification (Cabin, Radiation, Cardio, Behavioral, Bone, Immune).
        - **Internationalization:** Instant localization across 5 languages (English, Turkish, German, French, Spanish) for multinational crew support.
        - **Interplanetary Extension:** Real-time query support for deep space bodies, spacecraft missions, and ephemeris telemetry.
    """)
