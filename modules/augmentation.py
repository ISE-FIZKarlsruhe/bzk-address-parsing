import re
import json
import random
import requests
import pandas as pd
from pathlib import Path

GEONAMES_BASE = "http://api.geonames.org"
_CACHE_FILE   = Path(__file__).parent / ".geonames_cache.json"


# GeoNames client 
class GeoNamesLookup:
    """Thin wrapper around the GeoNames JSON API with disk caching."""

    def __init__(self, username: str, cache_file: Path = _CACHE_FILE):
        self.username = username
        self._cache_file = cache_file
        self._cache: dict = {}
        if cache_file.exists():
            try:
                with open(cache_file, encoding="utf-8") as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                print(f"[warning] cache file {cache_file} is corrupted, starting fresh")
                cache_file.unlink()

    def _get(self, endpoint: str, params: dict) -> dict:
        params = {**params, "username": self.username}
        cache_key = endpoint + "|" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        if cache_key not in self._cache:
            resp = requests.get(f"{GEONAMES_BASE}/{endpoint}", params=params, timeout=10)
            resp.raise_for_status()
            self._cache[cache_key] = resp.json()
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False)
        return self._cache[cache_key]

    def cities(self, country_code: str, max_rows: int = 1000) -> list[str]:
        """Return city names for a country, ordered by population (most populous first)."""
        data = self._get("searchJSON", {
            "country":       country_code,
            "featureClass":  "P",
            "maxRows":       max_rows,
            "orderby":       "population",
            "style":         "SHORT",
        })
        return [g["name"] for g in data.get("geonames", []) if g.get("name")]

    def states(self, country_code: str) -> list[str]:
        """Return first-level administrative division names."""
        data = self._get("searchJSON", {
            "country":       country_code,
            "featureClass":  "A",
            "featureCode":   "ADM1",
            "maxRows":       100,
            "style":         "SHORT",
        })
        return [g["name"] for g in data.get("geonames", []) if g.get("name")]



# ── Substitution helpers

def _replace_value(address: str, old: str, new: str) -> str | None:
    """
    Replace the labelled component old with new inside address.

    Uses a word-boundary-aware regex so that "Berlin" inside "Berliner Straße"
    is not accidentally replaced.  Returns None if old is not found as a
    standalone token sequence.
    """
    # Build a pattern that matches the old value at word boundaries.
    # re.escape handles special chars (dots, hyphens) in place names.
    pattern = r'(?<![^\W_])' + re.escape(old) + r'(?![^\W_])'
    if not re.search(pattern, address, re.UNICODE):
        return None
    return re.sub(pattern, new, address, count=1, flags=re.UNICODE)


# ── Core augmentation 

def augment_row(
    row:        pd.Series,
    city_pool:  list[str],
    rng:        random.Random,
    n_augments: int = 3,
) -> list[pd.Series]:
    """
    Produce up to n new rows by substituting City with a randomly
    sampled alternative from the GeoNames pool.  State is left unchanged.
    """
    results = []
    old_city = row.get("City")
    if not (city_pool and pd.notna(old_city) and old_city != ""):
        return results

    for _ in range(n_augments):
        new_city = rng.choice(city_pool)
        if new_city == old_city:
            continue
        replaced = _replace_value(row["FullAddress"], old_city, new_city)
        if replaced is None:
            continue
        new_row = row.copy()
        new_row["FullAddress"]    = replaced
        new_row["City"]           = new_city
        new_row["is_augmented"]   = True
        new_row["source_address"] = row["FullAddress"]
        results.append(new_row)

    return results


# Mapping from Country column values → ISO-3166 alpha-2 codes.
# Covers full names, common abbreviations, German-language names/abbreviations,
# and historical states (mapped to their modern successors).
# Extend via the `extra_country_map` parameter of augment_dataset.
COUNTRY_TO_ISO: dict[str, str] = {
    # ── Modern countries – ISO codes ──────────────────────────────────────────
    "DE": "DE", "AT": "AT", "CH": "CH", "PL": "PL", "FR": "FR",
    "GB": "GB", "US": "US", "RU": "RU", "HU": "HU", "RO": "RO",
    "BG": "BG", "CZ": "CZ", "SK": "SK", "HR": "HR", "RS": "RS",
    "UA": "UA", "BY": "BY", "LT": "LT", "LV": "LV", "EE": "EE",
    "NL": "NL", "BE": "BE", "IT": "IT", "ES": "ES", "PT": "PT",
    "GR": "GR", "TR": "TR", "IL": "IL",
    # ── German full names ──────────────────────────────────────────────────────
    "Deutschland": "DE", "Germany": "DE",
    "Österreich": "AT", "Austria": "AT",
    "Schweiz": "CH", "Switzerland": "CH",
    "Polen": "PL", "Poland": "PL",
    "Frankreich": "FR", "France": "FR",
    "England": "GB", "Großbritannien": "GB", "Great Britain": "GB",
    "Ungarn": "HU", "Hungary": "HU",
    "Rumänien": "RO", "Romania": "RO",
    "Bulgarien": "BG", "Bulgaria": "BG",
    "Tschechien": "CZ", "Czechia": "CZ", "Czech Republic": "CZ",
    "Slowakei": "SK", "Slovakia": "SK",
    "Kroatien": "HR", "Croatia": "HR",
    "Serbien": "RS", "Serbia": "RS",
    "Ukraine": "UA",
    "Weißrussland": "BY", "Belarus": "BY",
    "Litauen": "LT", "Lithuania": "LT",
    "Lettland": "LV", "Latvia": "LV",
    "Estland": "EE", "Estonia": "EE",
    "Niederlande": "NL", "Netherlands": "NL", "Holland": "NL",
    "Belgien": "BE", "Belgium": "BE",
    "Italien": "IT", "Italy": "IT",
    "Spanien": "ES", "Spain": "ES",
    "Portugal": "PT",
    "Griechenland": "GR", "Greece": "GR",
    "Türkei": "TR", "Turkey": "TR",
    "Russland": "RU", "Russia": "RU",
    "Israel": "IL",
    "USA": "US", "United States": "US",
    # ── German abbreviations (with and without dot) ────────────────────────────
    "Dtschl.": "DE", "Dtschl": "DE", "Dtl.": "DE", "Dtl": "DE",
    "Öst.": "AT", "Öst": "AT",
    "Schw.": "CH",                          # Schweiz
    "Pol.": "PL", "Pol": "PL",
    "Ung.": "HU", "Ung": "HU",              # Ungarn
    "Rum.": "RO", "Rum": "RO",              # Rumänien
    "Bulg.": "BG", "Bulg": "BG",            # Bulgarien
    "Tsch.": "CZ", "Tsch": "CZ",            # Tschechien / Tschechoslowakei
    "Jug.": "RS", "Jug": "RS",              # Jugoslawien → Serbia as main successor
    "Ukr.": "UA", "Ukr": "UA",
    "Lit.": "LT", "Lit": "LT",
    "Lett.": "LV", "Lett": "LV",
    "Niederl.": "NL", "Niederl": "NL",
    "Belg.": "BE", "Belg": "BE",
    "It.": "IT",
    "Span.": "ES", "Span": "ES",
    "Griech.": "GR", "Griech": "GR",
    "Türk.": "TR", "Türk": "TR",
    "Russ.": "RU", "Russ": "RU",
    "Isr.": "IL", "Isr": "IL",
    # ── Historical states (mapped to modern successors) ────────────────────────
    # Czechoslovakia
    "CSR": "CZ", "ČSR": "CZ", "CSSR": "CZ", "ČSSR": "CZ",
    "Tschechoslowakei": "CZ", "Czechoslovakia": "CZ",
    # Soviet Union
    "UdSSR": "RU", "UDSSR": "RU", "USSR": "RU", "SU": "RU",
    "Sowjetunion": "RU", "Soviet Union": "RU",
    # Yugoslavia
    "Jugoslawien": "RS", "Yugoslavia": "RS", "SFRJ": "RS",
    # German states / empires
    "Deutsches Reich": "DE", "DR": "DE",
    "DDR": "DE", "Deutsche Demokratische Republik": "DE",
    "Preußen": "DE", "Prussia": "DE",
    "Österreich-Ungarn": "AT", "Austria-Hungary": "AT"
}


def augment_dataset(
    df:                pd.DataFrame,
    geo:               GeoNamesLookup,
    n_augments:        int = 3,
    seed:              int = 42,
    extra_country_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Augment every row in df that has a City or State label, using the row's
    own Country column to determine which GeoNames pool to sample from.

    Rows whose country is not found in COUNTRY_TO_ISO are silently skipped.
    """
    # Build a lowercased version of the map for case-insensitive lookup
    iso_map = {k.lower(): v for k, v in {**COUNTRY_TO_ISO, **(extra_country_map or {})}.items()}

    # Pre-fetch city pools once per unique resolved ISO code
    country_pools: dict[str, list[str]] = {}
    for raw_country in df["Country"].dropna().unique():
        if raw_country == "":
            continue
        iso = iso_map.get(raw_country.lower())
        if iso is None:
            continue  # unknown country – skip silently
        if iso not in country_pools:
            print(f"Fetching city pool for {raw_country!r} ({iso}) …")
            cities = geo.cities(iso)
            print(f"  → {len(cities)} cities")
            country_pools[iso] = cities

    rng = random.Random(seed)
    augmented_rows = []

    for _, row in df.iterrows():
        if not (pd.notna(row.get("City")) and row.get("City") != ""):
            continue

        raw_country = row.get("Country", "")
        iso = iso_map.get(raw_country.lower()) if pd.notna(raw_country) and raw_country != "" else None
        if iso is None or iso not in country_pools:
            continue

        augmented_rows.extend(augment_row(row, country_pools[iso], rng, n_augments))

    # Mark original rows
    originals = df.copy()
    originals["is_augmented"]   = False
    originals["source_address"] = pd.NA

    if not augmented_rows:
        return originals.reset_index(drop=True)

    synthetic = pd.DataFrame(augmented_rows).reset_index(drop=True)
    return pd.concat([originals, synthetic], ignore_index=True)
