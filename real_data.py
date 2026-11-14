# -*- coding: utf-8 -*-
"""
real_data.py — opsiyonel GERÇEK NASA verisi yardımcıları (internet gerektirir)

  DONKI SEP  : güneş enerjik parçacık olayları (radyasyon riski)  https://ccmc.gsfc.nasa.gov/DONKI-API/get/SEP
  OSDR search: spaceflight biyoloji çalışmaları (kemik, kalp, bağışıklık, radyasyon)  https://osdr.nasa.gov/osdr/data/search

Kullanım:
  python src/real_data.py --donki 2026-09-15 2026-11-13     # SEP olayları (aralık <= 60 gün)
  python src/real_data.py --osdr "bone loss"                  # çalışma arama
  python src/real_data.py --selftest                          # internetsiz ayrıştırıcı testi
"""
import argparse
from typing import List

import pandas as pd

DONKI_SEP = "https://ccmc.gsfc.nasa.gov/DONKI-API/get/SEP"
OSDR_SEARCH = "https://osdr.nasa.gov/osdr/data/search"


def parse_donki_sep(events: List[dict]) -> pd.DataFrame:
    rows = []
    for e in events or []:
        instruments = ", ".join(i.get("displayName", "") for i in (e.get("instruments") or []) if isinstance(i, dict))
        rows.append({"event_time_utc": e.get("eventTime"), "sep_id": e.get("sepID"),
                     "instruments": instruments, "link": e.get("link")})
    df = pd.DataFrame(rows, columns=["event_time_utc", "sep_id", "instruments", "link"])
    if not df.empty:
        df["event_time_utc"] = pd.to_datetime(df["event_time_utc"], utc=True, errors="coerce")
        df = df.sort_values("event_time_utc").reset_index(drop=True)
    return df


def fetch_donki_sep(start, end, timeout=30) -> pd.DataFrame:
    import requests  # geç import: ayrıştırıcı testi requests gerektirmez
    r = requests.get(DONKI_SEP, params={"startDate": str(start), "endDate": str(end)}, timeout=timeout)
    r.raise_for_status()
    return parse_donki_sep(r.json())


def parse_osdr_hits(payload: dict) -> pd.DataFrame:
    hits = (payload or {}).get("hits", {})
    rows = []
    for h in hits.get("hits", []):
        src = h.get("_source", {})
        rows.append({"accession": src.get("Accession"), "title": src.get("Study Title")})
    return pd.DataFrame(rows, columns=["accession", "title"])


def fetch_osdr_studies(term, size=8, timeout=60) -> pd.DataFrame:
    import requests
    r = requests.get(OSDR_SEARCH, params={"term": term, "type": "cgene,alsda", "from": 0, "size": size}, timeout=timeout)
    r.raise_for_status()
    return parse_osdr_hits(r.json())


def _selftest():
    ev = parse_donki_sep([
        {"eventTime": "2026-10-02T10:05Z", "sepID": "A", "instruments": [{"displayName": "GOES-19: SEPS"}], "link": "x"},
        {"eventTime": "2026-09-20T01:00Z", "sepID": "B", "instruments": []},
    ])
    assert list(ev["sep_id"]) == ["B", "A"], ev          # zamana göre sıralı
    assert ev.loc[1, "instruments"] == "GOES-19: SEPS"
    assert parse_donki_sep([]).empty
    st = parse_osdr_hits({"hits": {"total": 1, "hits": [{"_source": {"Accession": "OSD-1", "Study Title": "T"}}]}})
    assert st.loc[0, "accession"] == "OSD-1" and parse_osdr_hits({}).empty
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--donki", nargs=2, metavar=("START", "END"))
    ap.add_argument("--osdr", metavar="TERM")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    if a.donki:
        print(fetch_donki_sep(*a.donki).to_string(index=False))
    if a.osdr:
        print(fetch_osdr_studies(a.osdr).to_string(index=False))
