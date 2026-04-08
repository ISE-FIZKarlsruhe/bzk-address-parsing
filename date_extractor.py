
import json, re
import logging
import pandas as pd
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)
from typing import Optional
import pandas as pd



GERMAN_MONTHS = {
    'januar': 1,  'jan': 1,   'january': 1,  'j': 1,
    'jänner': 1,  'janner': 1,
    'februar': 2, 'feb': 2,   'febr': 2,     'february': 2,
    'maerz': 3,   'märz': 3,  'mar': 3,      'mär': 3,    'march': 3,
    'april': 4,   'apr': 4,
    'mai': 5,     'may': 5,
    'juni': 6,    'jun': 6,   'june': 6,     'jani': 6,
    'juli': 7,    'jul': 7,   'july': 7,
    'august': 8,  'aug': 8,   'ug': 8,       'augl': 8,
    'september': 9, 'sep': 9, 'sept': 9,
    'oktober': 10, 'okt': 10, 'oct': 10,     'october': 10, 'k': 10, 'ct':10,
    'november': 11, 'nov': 11, 'novl': 11,
    'dezember': 12, 'dez': 12, 'dec': 12,    'december': 12, 'dezemb': 12,
    'elul': 6,   
}

ROMAN = {
    'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5, 'vi': 6,
    'vii': 7, 'viii': 8, 'ix': 9, 'x': 10, 'xi': 11, 'xii': 12,
    'xiii': 13,  
}

NON_DATE_TOKENS = {
    'unbekannt', 'deportiert', 'verstorben', 'verst', 'deportation',
    'umgekommen', 'gestorben', 'beide', 'unk',
}

#TODO: discuss mappings
SEASON_TO_MONTH = {'sommer': 8, 'frühling': 4, 'herbst': 10, 'winter': 1}

_EMPTY_RAW = {'nan', 'none', 'null', '', '-', '?'}

_DATE_TOKEN = re.compile(r'\b(\d{1,2})[.\-/]+\s*(\d{1,2})[.\-/]+\s*(\d{2,4})\b')

# Two 4-digit years joined by connector — take the latest
_ODER_PAT = re.compile(
    r'\b(\d{4})\s*(?:oder|or|bis|und|,)\s*(\d{4})\b', re.IGNORECASE
)

# OCR digit-for-letter fixes applied before all pattern matching
_OCR_FIXES = [
    (re.compile(r'\b0([Kk][Tt])\b'), r'O\1'),   # 0kt → Okt
    (re.compile(r'\b1([Vv])\b'),     r'I\1'),   # 1V  → IV
    (re.compile(r'\b8([Uu][Ll])\b'), r'J\1'),   # 8ul → Jul
]

_UML = r'A-Za-z\u00c4\u00e4\u00d6\u00f6\u00dc\u00fc\u00df'


# Year each layout class started to be used.
# These are placeholders (all set to 1946) to be confirmed with archivists.
# The value is used to compute the age a person born in the 1800s would have
# reached by that year, which determines the plausibility of the 1800s century.
# _OFFICE_START_YEAR = {
#     # Frühe Phase — earliest claims?
#     "BY-HB-Frühe Phase":                          1946,
#     "HE-Frühe-Phase":                             1946,
#     "NI-Frühe-Phase":                             1946,
#     "NRW-Frühe-Phase":                            1946,
#     "RLP-Frühe-Phase":                            1946,
#     # Hauptphase 
#     "BW-Hauptphase":                              1946,
#     "BY-BE-Hauptphase":                           1946,
#     "HE-Hauptphase":                              1946,
#     "HH-NI-NRW-SH-Hauptphase":                   1946,
#     "HH-NI-NRW-SH-Hauptphase (abweichend 1)":    1946,
#     "HH-NI-NRW-SH-Hauptphase (abweichend 2)":    1946,
#     "HH-NI-NRW-SH-Hauptphase (abweichend 3)":    1946,
#     "HH-NI-NRW-SH-Hauptphase (abweichend 4)":    1946,
#     "RLP-Hauptphase":                             1946,
#     "RLP-Hauptphase (abweichend 1)":              1946,
#     "RLP-Hauptphase (abweichend 2)/Saarland":     1946,
#     "NRW-Köln-Art-V":                             1946,
#     "NRW-Köln-Härtefonds":                        1946,
#     "NRW-LRB":                                    1946,
#     "NRW-Innenministerium":                       1946,
#     # Spätphase — late claims?
#     "BY-Spätphase":                               1946,
#     # Special/index-card layouts
#     "ABC-Karten":                                 1946,
#     "Hinweiskarte (Mainz/Neustadt)":              1946,
#     "Hinweiskarte_schwarzes_Datum":               1946,
#     "Tabellen-Typ":                               1946,
#     "Tabellen-Typ (abweichend)":                  1946,
# }
# _DEFAULT_OFFICE_START = 1946   # fallback for unknown layout classes

# # Age thresholds (age of 1800s-born person at the reference year) that determine
# # the uncertainty level assigned to the century guess.
# _AGE_LEVEL_3 = 105   # age ≥ 105 → essentially impossible
# _AGE_LEVEL_2 = 93    # age ≥  93 → extremely unlikely    

# # Fixed reference year for VictimBirthDate: earliest year of Nazi persecution.
# # A victim must have been alive in 1933; the persecution era ran 1933-1945.
# _PERSECUTION_REF_YEAR = 1933

_MULTI_LABELED = re.compile(r'[ab1-9]\)\s*')
_TWO_DATES = re.compile(
    r'^(\d{1,2}[.\-/]+\d{1,2}[.\-/]+\d{2,4})\s+(\d{1,2}[.\-/]+\d{1,2}[.\-/]+\d{2,4})$'
)
_SKIP_RAW = {'nan', 'none', 'null', '', '-', '?'}

# Filenames must follow YYYY_MM_DD_seq_seq - others are non-person records
_FNAME_PAT = re.compile(r'^(\d{4})_(\d{2})_(\d{2})_\d+_\d+$')
_DATE_FIELDS  = ['ApplicantBirthDate', 'VictimBirthDate', 'VictimDeathDate']
_SKIP_RAW_SET = {'nan', 'none', 'null', '', '-', '?'}

OUTPUT_PATH = 'resolved_dates.jsonl'
_DATE_FIELDS = ['ApplicantBirthDate', 'VictimBirthDate', 'VictimDeathDate']


_LAYOUT_FILENAME_FIELD = {
    # filename = VictimBirthDate
    "BY-HB-Frühe Phase":               "VictimBirthDate",
    "BY-eigener-Typ (abweichend 1)":   "VictimBirthDate",
    "HH-Frühe-Phase":                  "VictimBirthDate",
    "HE-Frühe-Phase":                  "VictimBirthDate",  
    "NRW-Frühe-Phase":                 "VictimBirthDate",

    # filename = ApplicantBirthDate
    "BY-BE-Hauptphase":                            "ApplicantBirthDate",
    "BY-eigener-Typ":                              "ApplicantBirthDate",
    "BY-eigener-Typ (abweichend 2)":               "ApplicantBirthDate",
    "BY-Spätphase":                                "ApplicantBirthDate",
    "BY-Spätphase (abweichend 1)":                 "ApplicantBirthDate",
    "HH-NI-NRW-SH-Hauptphase":                    "ApplicantBirthDate",
    "HH-NI-NRW-SH-Hauptphase (abweichend 1)":     "ApplicantBirthDate",
    "HH-NI-NRW-SH-Hauptphase (abweichend 2)":     "ApplicantBirthDate",
    "HH-NI-NRW-SH-Hauptphase (abweichend 3)":     "ApplicantBirthDate",
    "HH-NI-NRW-SH-Hauptphase (abweichend 4)":     "ApplicantBirthDate",
    "NI-Frühe-Phase":                              "ApplicantBirthDate",   
    "Tabellen-Typ":                                "ApplicantBirthDate",
    "Tabellen-Typ (abweichend)":                   "ApplicantBirthDate",
    "HB-Spätphase":                                "ApplicantBirthDate",
    "HE-Hauptphase":                               "ApplicantBirthDate",
    "NRW-LRB":                                     "ApplicantBirthDate",
    "NRW-Köln-Art-V":                              "ApplicantBirthDate",
    "NRW-Köln-Härtefonds":                         "ApplicantBirthDate",
    "RLP-Hauptphase":                              "ApplicantBirthDate",
    "RLP-Hauptphase (abweichend 1)":               "ApplicantBirthDate",
    "RLP-Hauptphase (abweichend 2) / Saarland":    "ApplicantBirthDate",
    "RLP-Hauptphase (abweichend 3)":               "ApplicantBirthDate",
    "RLP-Hauptphase (abweichend 4)":               "ApplicantBirthDate",
    "Hinweiskarte_schwarzes_Datum":                "ApplicantBirthDate",
    "BW-Hauptphase":                               "ApplicantBirthDate",

    # No predefined role order = skip
    "NRW-Innenministerium":                           None,
    "Auskünfte_Statistisches_Landesamt_NRW":          None,
    "Auskünfte_Statistisches_Landesamt_NRW (abweichend)": None,
    "RLP-Frühe-Phase":                                None,
    "ABC-Karten":                                     None,
    "Suchkarte-Hinweiskarte":                         None,
}

# Fields structurally absent from specific layout classes
_LAYOUT_ABSENT_FIELDS = {
    # HE-Frühe-Phase: "kein Feld für das Geburtsdatum vorgesehen" for Antragstellende
    "HE-Frühe-Phase":      {"ApplicantBirthDate"},
    # NRW-LRB and NRW-Köln-Härtefonds: only Anspruchsberechtigter, no victim role
    "NRW-LRB":             {"VictimBirthDate"},
    "NRW-Köln-Härtefonds": {"VictimBirthDate"},
}

# Layout classes where the date of interest is VictimBirthDate
_VICTIM_BIRTHDATE_CLASSES = {
    "BY-HB-Frühe Phase",
    "HE-Frühe-Phase",
    "NRW-Frühe-Phase",
}

_IGNORE_CLASSES = {
    "Gerichtsurteile", "Gelbe-Hinweiskarte",
    "Rückseite_Weitere Namen", "Siehe-auch-Hinweiskarte",
}


_META_FILES = {
    "480":    Path("metadata/findbuch-480-bearbeitet.xlsx"),
    "EL-350": Path("metadata/findbuch-EL-350-I-bearbeitet.xlsx"),
    "F-196":  Path("metadata/findbuch-F-196-gesamt-bearbeitet.xlsx"),
    "Wue-33": Path("metadata/findbuch-Wue-33-T-1-bearbeitet.xlsx"),
}

BZK_DATA_DIR = Path("/home/bzk-data")


class _UncertainYear(int):
    """Int subclass marking a guessed century for an ambiguous 2-digit year.

    level: 1 = slightly uncertain, 2 = unlikely, 3 = very unlikely
    """
    def __new__(cls, value, level=1):
        obj = int.__new__(cls, value)
        obj.level = level
        return obj
    def __add__(self, other):
        return _UncertainYear(int.__add__(self, other), self.level)
    def __radd__(self, other):
        return _UncertainYear(int.__radd__(self, other), self.level)




def _ocr_fix(s):
    for pat, repl in _OCR_FIXES:
        s = pat.sub(repl, s)
    return s


def _parse_roman(s):
    """Parse any Roman numeral string (case-insensitive). Returns 0 on failure."""
    vals = {'i': 1, 'v': 5, 'x': 10, 'l': 50, 'c': 100}
    s = s.lower()
    if not s or not all(c in vals for c in s):
        return 0
    result, prev = 0, 0
    for ch in reversed(s):
        v = vals[ch]
        result += v if v >= prev else -v
        prev = v
    return result



def _extract_year(raw):
    """Extract a plausible year from a raw date string
    """
    if not raw:
        return None
    hit = re.search(r'\b(1[89]\d{2}|20\d{2})\b', raw)
    if hit:
        return int(hit.group(1))
    # 2-digit year at the end of a date pattern — treat as 1900s 
    hit = re.search(r'(?:^|[.\-/\s])(\d{2})$', raw.strip())
    if hit:
        return 1900 + int(hit.group(1))
    return None


def _century(yy, col, ctx=None):
    """Resolve a 2-digit year to a century using column name and optional context dict.

    The ambiguous ranges and unambiguous boundaries are defined by archival rules:
        ApplicantBirthDate: yy >= 55 → 1800s | yy <= 33 → 1900s | 34–54 ambiguous
        VictimBirthDate:    yy >= 45 → 1800s | yy <= 33 → 1900s | 34–44 ambiguous

    Within the ambiguous ranges, uncertainty levels are derived empirically from
    filename ground-truth and metadata birth-year distributions 

    VictimBirthDate also uses death_year cross-reference to resolve unambiguously
    when one century implies an implausible age at death (outside 0–105 years).
    """
    if 'DeathDate' in (col or ''):
        return 1900

    death_year = (ctx or {}).get('death_year')

    # VictimBirthDate: cross-reference with death year when available.
    if death_year is not None and col == 'VictimBirthDate':
        age_if_1900s = death_year - (1900 + yy)
        age_if_1800s = death_year - (1800 + yy)
        plausible_1900s = 0 <= age_if_1900s <= 105
        plausible_1800s = 0 <= age_if_1800s <= 105
        if plausible_1900s and not plausible_1800s:
            return 1900
        if plausible_1800s and not plausible_1900s:
            return 1800
        # Both plausible — fall through to field-specific heuristic

    if col == 'ApplicantBirthDate':
        if yy >= 55: return 1800                       # unambiguous 1800s
        if yy <= 33: return 1900                       # unambiguous 1900s
        # Ambiguous range 34–54 — empirical sub-divisions:
        if yy >= 50: return _UncertainYear(1900, 1)    # 36% of records are 1800s
        if yy >= 48: return _UncertainYear(1900, 2)    # 11% of records are 1800s
        return          _UncertainYear(1900, 3)        #  0% of records are 1800s

    if col == 'VictimBirthDate':
        if yy >= 45: return 1800                       # unambiguous 1800s
        if yy <= 33: return 1900                       # unambiguous 1900s
        # Ambiguous range 34–44 — empirical sub-divisions:
        if yy >= 43: return _UncertainYear(1900, 2)    #  2% of records are 1800s
        return          _UncertainYear(1900, 3)        #  0% of records are 1800s

    # Fallback
    return 1800 if yy >= 50 else 1900


def _fix_component(v):
    if v > 31 and v >= 10:
        rev = int(str(v)[::-1])
        if 1 <= rev <= 31:
            return rev
    return v


def _to_iso(day, month, year, partial=False):
    try:
        d, m, y = _fix_component(int(day)), int(month), int(year)
        m = _fix_component(m)
        if m > 12 and d <= 12:
            d, m = m, d
        if 1 <= m <= 12 and 1 <= d <= 31:
            return f'{y:04d}-{m:02d}-{d:02d}'
        if partial:
            return f'{y:04d}-{m:02d}-01' if 1 <= m <= 12 else f'{y:04d}-01-01'
        return '', 0
    except (ValueError, TypeError):
        if partial:
            try: return f'{int(year):04d}-01-01'
            except: pass
        return '', 0


def _find_all_date_tokens(s):
    return [(h.group(1), h.group(2), h.group(3)) for h in _DATE_TOKEN.finditer(s)]


def _resolve_year(yr_str, col, ctx=None):
    yr_int = int(yr_str)
    if yr_int >= 100:
        return yr_int
    century = _century(yr_int, col, ctx)
    return century + yr_int


def _month_from_str(s):
    key = s.lower().rstrip('.')
    return GERMAN_MONTHS.get(key) or SEASON_TO_MONTH.get(key)


def _normalize_single(s, col=None, ctx=None):
    s = s.strip()

    # Unwrap if entire string is parenthesized: "(1885-06-07)"
    if s.startswith('(') and s.endswith(')') and '(' not in s[1:-1]:
        return _normalize_single(s[1:-1].strip(), col, ctx)

    # Preserve year in inline parens before stripping them: "25. Aug. (1920)"
    paren_yr = re.search(r'\((\d{4})\)', s)
    s = re.sub(r'\s*\([^)]*\)', '', s).strip()
    if paren_yr and not re.search(r'\b\d{4}\b', s):
        s = s.rstrip('.').strip() + ' ' + paren_yr.group(1)

    s = re.sub(r'\.$', '', s).strip()
    if not s or s.lower() in _EMPTY_RAW:
        return '', 0

    s = _ocr_fix(s)

    # Already ISO
    hit = re.match(r'^(\d{4})-(\d{2})-(\d{2})', s)
    if hit:
        return f'{hit.group(1)}-{hit.group(2)}-{hit.group(3)}'

    lower = s.lower()

    # Non-date status tokens
    if any(tok in lower for tok in NON_DATE_TOKENS):
        tokens = _find_all_date_tokens(s)
        if tokens:
            d, mo, yr = tokens[0]
            resolved = _resolve_year(yr, col, ctx)
            return _to_iso(d, mo, (resolved.level if isinstance(resolved, _UncertainYear) else 0), resolved)
        month_map = {**GERMAN_MONTHS, **SEASON_TO_MONTH}
        for mon_str, mo in month_map.items():
            if mon_str in lower:
                hit2 = re.search(r'\b(\d{4})\b', s)
                if hit2: return f'{hit2.group(1)}-{mo:02d}-01'
        return '', 0

    if s == '?':
        return '', 0
    hit = re.match(r'^\?\.\s*(\d{4})$', s)
    if hit:
        return f'{hit.group(1)}-01-01'

    # Two 4-digit years joined by connector — take the latest
    hit = _ODER_PAT.search(s)
    if hit:
        return f'{max(int(hit.group(1)), int(hit.group(2))):04d}-01-01'

    # "YYYY u. rest" — strip leading year+connector, normalize remainder
    hit = re.match(r'^\d{4}\s+u\.\s+(.+)$', s)
    if hit:
        return _normalize_single(hit.group(1), col)

    # "YYYY, rest" — leading year with comma; normalize rest with year injected
    hit = re.match(r'^(\d{4}),\s*(.+)$', s)
    if hit:
        yr, rest = hit.group(1), hit.group(2).strip().rstrip('.')
        hit2 = re.match(r'^(\d{1,2})[.\-/,]+(\d{1,2})$', rest)
        if hit2:
            resolved = _resolve_year(yr, col, ctx)
            return _to_iso(hit2.group(1), hit2.group(2), resolved, partial=True), (resolved.level if isinstance(resolved, _UncertainYear) else 0)
        candidate = _normalize_single(f'{rest} {yr}', col)
        if candidate:
            return candidate

    # "Jahr" keyword — date context marker; extract year
    if re.search(r'\bjahr\b', lower):
        hit = re.search(r'\b(\d{4})\b', s)
        if hit: return f'{hit.group(1)}-01-01'

    # D.M.YYYY or D.M.YY — accepts . - / , as separators
    hit = re.match(r'^(\d{1,2})[.\-/,]+\s*(\d{1,2})[.\-/,]+\s*(\d{2,4})$', s)
    if hit:
        resolved = _resolve_year(hit.group(3), col, ctx)
        return _to_iso(hit.group(1), hit.group(2), resolved, partial=True), (resolved.level if isinstance(resolved, _UncertainYear) else 0)

    # D M. YYYY — space before numeric month, separator after (also handles oversized day)
    hit = re.match(r'^(\d+)\s+(\d{1,2})[.\-]+\s*(\d{2,4})$', s)
    if hit:
        d_raw = int(hit.group(1))
        if d_raw > 31:
            last2 = int(str(d_raw)[-2:])
            day = last2 if 1 <= last2 <= 31 else 1
        else:
            day = d_raw
        resolved = _resolve_year(hit.group(3), col, ctx)
        return _to_iso(day, hit.group(2), resolved, partial=True), (resolved.level if isinstance(resolved, _UncertainYear) else 0)

    # Oversized day (3-4 digits).M.YYYY — take last 2 digits of first component
    hit = re.match(r'^(\d{3,4})[.\-/]+(\d{1,2})[.\-/]+(\d{4})$', s)
    if hit:
        last2 = int(hit.group(1)[-2:])
        return _to_iso(last2 if 1 <= last2 <= 31 else 1, hit.group(2), hit.group(3), partial=True)

    # YYYY.M.D
    hit = re.match(r'^(\d{4})[.\-/]+(\d{1,2})[.\-/]+(\d{1,2})$', s)
    if hit:
        return _to_iso(hit.group(3), hit.group(2), hit.group(1), partial=True)

    # MM/DD/YYYY US
    hit = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', s)
    if hit:
        return _to_iso(hit.group(2), hit.group(1), hit.group(3))

    # D.MYY — merged: "3.275" = day=3, month=2, year=75
    hit = re.match(r'^(\d{1,2})\.(\d)(\d{2})$', s)
    if hit:
        resolved = _resolve_year(hit.group(3), col, ctx)
        return _to_iso(hit.group(1), hit.group(2), resolved, partial=True), (resolved.level if isinstance(resolved, _UncertainYear) else 0)

    # D.Roman.YYYY — e.g. "16.XIII.00" → 1900-12-16
    hit = re.match(r'^(\d{1,2})[\s.\-/,]+([IVXivxLlCc]+)[\s.\-/,]+(\d{2,4})$', s)
    if hit:
        mo_r = _parse_roman(hit.group(2))
        if mo_r:
            resolved = _resolve_year(hit.group(3), col, ctx)
            return _to_iso(hit.group(1), min(mo_r, 12), resolved, partial=True), (resolved.level if isinstance(resolved, _UncertainYear) else 0)

    # Roman.Roman.YYYY — e.g. "II.II.1940" → 1940-02-02
    hit = re.match(r'^([IVXivx]+)[.\-/]+([IVXivx]+)[.\-/]+(\d{2,4})$', s)
    if hit:
        day_r = _parse_roman(hit.group(1))
        mo_r  = _parse_roman(hit.group(2))
        if day_r and mo_r:
            resolved = _resolve_year(hit.group(3), col, ctx)
            return _to_iso(day_r, mo_r, (resolved.level if isinstance(resolved, _UncertainYear) else 0), resolved)

    # Roman. D. YYYY — Roman=month, numeric=day: "II. 6. 1936" → 1936-02-06
    hit = re.match(r'^([IVXivx]+)[.\-\s]+(\d{1,2})[.\-\s]+(\d{2,4})$', s)
    if hit:
        mo_r = _parse_roman(hit.group(1))
        if mo_r:
            resolved = _resolve_year(hit.group(3), col, ctx)
            return _to_iso(hit.group(2), mo_r, (resolved.level if isinstance(resolved, _UncertainYear) else 0), resolved)

    # Roman. Month YYYY — Roman=day, text=month: "II. Sept. 1937" → 1937-09-02
    hit = re.match(rf'^([IVXivx]+)[.\-\s]+([{_UML}]+)\.?[.\-\s,]*(\d{{2,4}})$', s)
    if hit:
        day_r = _parse_roman(hit.group(1))
        mo = _month_from_str(hit.group(2))
        if day_r and mo:
            resolved = _resolve_year(hit.group(3), col, ctx)
            return _to_iso(day_r, mo, (resolved.level if isinstance(resolved, _UncertainYear) else 0), resolved)

    # D1/D2 M. YYYY — two alternative days, numeric month: "2/8 3. 1930" → 1930-03-02
    hit = re.match(r'^(\d{1,2})/\d+\s+(\d{1,2})[.\-]+\s*(\d{2,4})$', s)
    if hit:
        mo = int(hit.group(2))
        if 1 <= mo <= 12:
            resolved = _resolve_year(hit.group(3), col, ctx)
            return _to_iso(hit.group(1), mo, (resolved.level if isinstance(resolved, _UncertainYear) else 0), resolved)

    # D1/D2. Month YYYY — two alternative days, text month: "14/26. März 1880" → 1880-03-14
    hit = re.match(rf'^(\d{{1,2}})/\d{{1,2}}[.\s]+([{_UML}]+)\.?[.\-\s,]*(\d{{2,4}})$', s)
    if hit:
        mo = _month_from_str(hit.group(2))
        if mo:
            resolved = _resolve_year(hit.group(3), col, ctx)
            return _to_iso(hit.group(1), mo, (resolved.level if isinstance(resolved, _UncertainYear) else 0), resolved)

    # D/D YYYY — ambiguous fraction; extract year only: "5/17 1944" → 1944-01-01
    hit = re.match(r'^(\d{1,2})/(\d{1,2})\s+(\d{4})$', s)
    if hit:
        return f'{hit.group(3)}-01-01'

    # Ordinal day + Month + YYYY: "19th March 1913" → 1913-03-19
    hit = re.match(rf'^(\d{{1,2}})(?:st|nd|rd|th)[.,\s]+([{_UML}]+)\.?\s*(\d{{4}})$', s)
    if hit:
        mo = _month_from_str(hit.group(2))
        if mo:
            resolved = _resolve_year(hit.group(3), col, ctx)
            return _to_iso(hit.group(1), mo, (resolved.level if isinstance(resolved, _UncertainYear) else 0), resolved)

    # Month oder Month YYYY — take the second (later) month
    hit = re.match(rf'^([{_UML}]+)\.?\s+(?:oder|or)\s+([{_UML}]+)\.?\s*(\d{{4}})$', s)
    if hit:
        mo = _month_from_str(hit.group(2))
        if mo: return f'{hit.group(3)}-{mo:02d}-01'

    # D Month YYYY — includes / in after-month separator for cases like "1.Jul/1883"
    hit = re.match(
        rf'^(\d{{1,2}})[.\-\s,]+([{_UML}]+)\.?[.\-\s,/]*(\d{{2,4}})$', s
    )
    if hit:
        mo = _month_from_str(hit.group(2))
        if mo:
            resolved = _resolve_year(hit.group(3), col, ctx)
            return _to_iso(hit.group(1), mo, resolved, partial=True), (resolved.level if isinstance(resolved, _UncertainYear) else 0)

    # Im/In Month YYYY
    hit = re.match(rf'^[Ii][mn]\s+([{_UML}]+)\.?\s*(\d{{4}})$', s)
    if hit:
        mo = _month_from_str(hit.group(1))
        if mo: return f'{hit.group(2)}-{mo:02d}-01'

    # Month D, YYYY
    hit = re.match(rf'^([{_UML}]+)\.?\s+(\d{{1,2}}),?\s*(\d{{4}})$', s)
    if hit:
        mo = _month_from_str(hit.group(1))
        if mo: return _to_iso(hit.group(2), mo, hit.group(3))

    # Month YYYY or YY — "Feb. 42", "August 1942", "Augl 1901"
    hit = re.match(rf'^([{_UML}]+)\.?\s*(\d{{2,4}})$', s)
    if hit:
        mo = _month_from_str(hit.group(1))
        if mo:
            resolved = _resolve_year(hit.group(2), col, ctx)
            return f'{resolved:04d}-{mo:02d}-01', (resolved.level if isinstance(resolved, _UncertainYear) else 0)

    # YYYY D. Month — day before text month: "1889 21. Elul"
    hit = re.match(rf'^(\d{{4}})\s+(\d{{1,2}})[.\s]+([{_UML}]+)\.?$', s)
    if hit:
        mo = _month_from_str(hit.group(3))
        if mo: return _to_iso(hit.group(2), mo, hit.group(1))

    # YYYY Month YYYY — spurious leading year: "1940 März 1939" → 1939-03-01
    hit = re.match(rf'^(\d{{4}})\s+([{_UML}]+)\.?\s*(\d{{4}})$', s)
    if hit:
        mo = _month_from_str(hit.group(2))
        if mo: return f'{hit.group(3)}-{mo:02d}-01'

    # YYYY Month D — "1894 Aug. 18"
    hit = re.match(rf'^(\d{{4}})\s+([{_UML}]+)\.?\s*(\d{{1,2}})$', s)
    if hit:
        mo = _month_from_str(hit.group(2))
        if mo: return _to_iso(hit.group(3), mo, hit.group(1))

    # YYYY. Roman. D — "1897. XII. 23"
    hit = re.match(r'^(\d{4})[.\-\s]+([IVXivx]+)[.\-\s]+(\d{1,2})$', s)
    if hit:
        mo_r = _parse_roman(hit.group(2))
        if mo_r: return _to_iso(hit.group(3), min(mo_r, 12), hit.group(1))

    # YYYY/Month or YYYY Month — "1882/November", "1875 März"
    hit = re.match(rf'^(\d{{4}})[/\s]+([{_UML}]+)\.?$', s)
    if hit:
        mo = _month_from_str(hit.group(2))
        if mo: return f'{hit.group(1)}-{mo:02d}-01'

    # YYYY-Month text — "1903-Sept."
    hit = re.match(rf'^(\d{{4}})[.\-]([{_UML}]+)\.?$', s)
    if hit:
        mo = _month_from_str(hit.group(2))
        if mo: return f'{hit.group(1)}-{mo:02d}-01'

    # YYYY im Month — "1923 im April"
    hit = re.match(rf'^(\d{{4}})\s+[Ii]m\s+([{_UML}]+)\.?$', s)
    if hit:
        mo = _month_from_str(hit.group(2))
        if mo: return f'{hit.group(1)}-{mo:02d}-01'

    # Ende YYYY
    hit = re.match(r'^[Ee]nde\s+(\d{4})$', s)
    if hit: return f'{hit.group(1)}-12-01'

    # M/YYYY or M.YYYY — "11/1906" → 1906-11-01
    hit = re.match(r'^(\d{1,2})[./](\d{4})$', s)
    if hit:
        m = int(hit.group(1))
        if 1 <= m <= 12: return f'{hit.group(2)}-{m:02d}-01'

    # YYYY-MM partial
    hit = re.match(r'^(\d{4})[.\-/](\d{1,2})$', s)
    if hit:
        m = int(hit.group(2))
        if 1 <= m <= 12: return f'{hit.group(1)}-{m:02d}-01'

    # YYYY only
    hit = re.match(r'^(\d{4})$', s)
    if hit: return f'{hit.group(1)}-01-01'

    # M. YY or M. YYYY: "11. 1859", "12.75", "2.1906"
    hit = re.match(r'^(\d{1,2})\.\s*(\d{2}|\d{4})$', s)
    if hit:
        resolved = _resolve_year(hit.group(2), col, ctx)
        return f'{resolved:04d}-{int(hit.group(1)):02d}-01', (resolved.level if isinstance(resolved, _UncertainYear) else 0)

    # YYYY-YYYY year range: take first
    hit = re.match(r'^(\d{4})[.\-/]+(\d{4})$', s)
    if hit: return f'{hit.group(1)}-01-01'

    # D.D. Month YYYY — strip leading double-numeric OCR noise: "19.36. Nov. 1936"
    hit = re.match(rf'^\d+[./]+\d+[./\s]+([{_UML}]+)\.?\s*(\d{{2,4}})$', s)
    if hit:
        mo = _month_from_str(hit.group(1))
        if mo:
            resolved = _resolve_year(hit.group(2), col, ctx)
            return f'{resolved:04d}-{mo:02d}-01', (resolved.level if isinstance(resolved, _UncertainYear) else 0)

    # Last resort: first D.M.Y token
    tokens = _find_all_date_tokens(s)
    if tokens:
        d, mo, yr = tokens[0]
        resolved = _resolve_year(yr, col, ctx)
        return _to_iso(d, mo, resolved, partial=True), (resolved.level if isinstance(resolved, _UncertainYear) else 0)

    # Year-only fallback: extract any plausible 4-digit year from noisy strings
    # e.g. "2. M. 1882", "ca. 1900", "187.1901", "22.1-4/2 1911"
    hit = re.search(r'\b(\d{4})\b', s)
    if hit:
        yr = int(hit.group(1))
        if 1800 <= yr <= 2024:
            return f'{yr:04d}-01-01'

    return ''


def _as_pair(x):
    """Ensure _normalize_single output is always a (iso_str, level) tuple."""
    if isinstance(x, tuple):
        return x[0] or '', x[1] if len(x) > 1 else 0
    return x or '', 0


def normalize_date(raw, col=None, ctx=None):
    """Normalise raw date string to ISO 8601. Multiple dates -> 'D1 ; D2'.
    """
    if not raw:
        return '', 0
    s = raw.strip()
    if not s or s.lower() in _EMPTY_RAW:
        return '', 0
    s_clean = re.sub(r'\s*\([^)]*\)', '', s).strip()
    if not s_clean:
        s_clean = s.strip('()').strip()

    parts = [p.strip() for p in _MULTI_LABELED.split(s_clean) if p.strip()]
    if len(parts) > 1:
        pairs = [_as_pair(_normalize_single(p, col, ctx)) for p in parts]
        results = [(iso, lvl) for iso, lvl in pairs if iso]
        if not results: return '', 0
        return ' ; '.join(iso for iso, _ in results), max(lvl for _, lvl in results)

    if ';' in s_clean:
        parts = [p.strip() for p in s_clean.split(';') if p.strip()]
        pairs = [_as_pair(_normalize_single(p, col, ctx)) for p in parts]
        results = [(iso, lvl) for iso, lvl in pairs if iso]
        if not results: return '', 0
        return ' ; '.join(iso for iso, _ in results), max(lvl for _, lvl in results)

    hit = _TWO_DATES.match(s_clean)
    if hit:
        pairs = [_as_pair(_normalize_single(hit.group(1), col, ctx)),
                 _as_pair(_normalize_single(hit.group(2), col, ctx))]
        results = [(iso, lvl) for iso, lvl in pairs if iso]
        if not results: return '', 0
        return ' ; '.join(iso for iso, _ in results), max(lvl for _, lvl in results)

    return _as_pair(_normalize_single(s, col, ctx))




def filename_to_gt(stem):
    m = _FNAME_PAT.match(stem)
    if not m:
        return None
    yyyy, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if mm == 0:
        return f"{yyyy:04d}-01-01"
    if dd == 0:
        return f"{yyyy:04d}-{mm:02d}-01"
    return f"{yyyy:04d}-{mm:02d}-{dd:02d}"

def extract_dates():
    log.info("Loading mapping CSV...")
    mapping_df = pd.read_csv(BZK_DATA_DIR / "matched_offices_by_fullname_2.csv")
    mapping_df = mapping_df[~mapping_df["Layout class"].isin(_IGNORE_CLASSES)].copy()

    by_year = defaultdict(dict)
    for _, row in mapping_df.iterrows():
        fname = row["Filename"]
        layout_class = row["Layout class"]
        stem = fname.rsplit(".", 1)[0]
        gt = filename_to_gt(stem)
        if gt is None:
            continue
        year = stem.split("_")[0]
        date_col = "VictimBirthDate" if layout_class in _VICTIM_BIRTHDATE_CLASSES else "ApplicantBirthDate"
        by_year[year][fname] = (gt, layout_class, date_col)

    n_files = sum(len(v) for v in by_year.values())
    log.info(f"Valid files in mapping: {n_files:,}  |  years: {min(by_year)} – {max(by_year)}")

    rows = []
    years = sorted(by_year.keys())
    for i, year in enumerate(years, 1):
        fname_map = by_year[year]
        jsonl = BZK_DATA_DIR / f"{year}.jsonl"
        if not jsonl.exists():
            continue
        log.info(f"  [{i}/{len(years)}] year {year}  ({len(fname_map):,} files to match)")
        with open(jsonl, encoding="utf-8") as fh:
            for line in fh:
                rec   = json.loads(line)
                fname = rec.get("filename", "")
                if fname not in fname_map:
                    continue
                gt, layout_class, date_col = fname_map[fname]
                raw  = (rec.get(date_col) or '').strip()
                death_raw  = (rec.get('VictimDeathDate') or '').strip()
                death_year = _extract_year(death_raw)
                pred, uncertainty_level = normalize_date(raw, col=date_col, ctx={'death_year': death_year}) if raw and raw.lower() not in _SKIP_RAW else ('', 0)
                rows.append({
                    "region":            layout_class,
                    "date_col":          date_col,
                    "filename":          fname,
                    "gt":                gt,
                    "raw":               raw,
                    "pred":              pred,
                    "correct":           pred == gt,
                    "uncertain":         pred.startswith('~'),
                    "uncertainty_level": uncertainty_level,
                    "empty":             pred == "" and bool(raw) and raw.lower() not in _SKIP_RAW,
                })

    regional_df = pd.DataFrame(rows)
    log.info(f"Records matched: {len(regional_df):,}  |  with raw value: {(regional_df['raw'] != '').sum():,}  |  no raw value: {(regional_df['raw'] == '').sum():,}")
    return regional_df



def eval_regional(df, label="ALL"):
    sub = df[df["raw"] != ""].copy()
    n           = len(sub)
    if n == 0:
        print(f"[{label}]  no records with raw date value")
        return
    n_correct   = sub["correct"].sum()
    n_uncertain = sub["uncertain"].sum()
    # Per-level uncertainty breakdown
    if "uncertainty_level" in sub.columns:
        n_u1 = (sub["uncertainty_level"] == 1).sum()
        n_u2 = (sub["uncertainty_level"] == 2).sum()
        n_u3 = (sub["uncertainty_level"] == 3).sum()
        unc_detail = f"uncertain(~={n_u1},~~={n_u2},~~~={n_u3})"
    else:
        unc_detail = f"uncertain(~)={n_uncertain:>5,}"
    n_empty     = sub["empty"].sum()
    print(
        f"[{label:<45}]  n={n:>7,}  "
        f"correct={n_correct:>7,} ({100*n_correct/n:5.1f}%)  "
        f"{unc_detail}  unparsed={n_empty:>5,}"
    )



def _norm_date_iso(s) -> Optional[str]:
    if pd.isna(s) or str(s).strip() in ('', 'NaN', 'nan'): return None
    s = str(s).strip()
    m = re.match(r'^(\d{4})-(\d{2})-(\d{2})', s)
    if m: return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.match(r'^(\d{1,2})\.(\d{1,2})\.(\d{4})', s)
    if m: return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    m = re.match(r'^(\d{4})$', s)
    if m: return f"{m.group(1)}-01-01"
    return None

def _norm_name(s) -> str:
    return str(s).strip().lower() if not pd.isna(s) else ''

def _load_metadata() -> pd.DataFrame:
    log.info("Loading metadata files...")
    dfs = []
    for src, path in _META_FILES.items():
        if not path.exists():
            log.warning(f'  [skip] {path} not found')
            continue
        log.info(f"  Reading {path.name}...")
        raw = pd.read_excel(path, header=None)
        df  = raw.iloc[2:].reset_index(drop=True)
        df.columns = raw.iloc[0].tolist()
        bd_col = (
            next((c for c in df.columns if 'geburt' in str(c).lower()
                  and 'datum' in str(c).lower() and 'format' in str(c).lower()), None)
            or next((c for c in df.columns if 'geburt' in str(c).lower()
                     and 'datum' in str(c).lower()), None)
        )
        sd_col = (
            next((c for c in df.columns if 'sterbe' in str(c).lower()
                  and 'datum' in str(c).lower() and 'format' in str(c).lower()), None)
            or next((c for c in df.columns if 'sterbe' in str(c).lower()
                     and 'datum' in str(c).lower()), None)
        )
        dfs.append(pd.DataFrame({
            'source':          src,
            'Bestellsignatur': df['Bestellsignatur'],
            'meta_last':       df['Nachname(n)'].apply(_norm_name),
            'meta_first':      df.get('Vorname(n)', pd.Series(dtype=str)).apply(_norm_name),
            'meta_bd':         (df[bd_col] if bd_col else pd.Series(dtype=str)).apply(_norm_date_iso),
            'meta_sd':         (df[sd_col] if sd_col else pd.Series(dtype=str)).apply(_norm_date_iso),
        }))
    meta = pd.concat(dfs, ignore_index=True)
    meta = meta[meta['meta_bd'].notna() & meta['meta_last'].ne('')]
    log.info(f"Metadata loaded: {len(meta):,} rows with name + birth date")
    return meta


def _build_merge_index(bzk_dir: Path, meta_df: pd.DataFrame) -> dict:
    """
    filename: {source, Bestellsignatur, meta_sd_iso}
    Joins on victim last name + birth date + first name (when both sides non-empty).
    Used only to recover VictimDeathDate from metadata; birth dates are excluded
    because the join key already includes birth date (circular recovery).
    """
    log.info("Building merge index: reading all JSONL files...")
    jsonl_files = sorted(bzk_dir.glob('*.jsonl'))
    rows = []
    for i, jsonl in enumerate(jsonl_files, 1):
        log.info(f"  [{i}/{len(jsonl_files)}] {jsonl.name}")
        with open(jsonl, encoding='utf-8') as fh:
            for line in fh:
                rec = json.loads(line)
                rows.append({
                    'filename':  rec.get('filename', ''),
                    'vic_last':  _norm_name(rec.get('VictimLastName')),
                    'vic_first': _norm_name(rec.get('VictimFirstName')),
                    'vic_bd':    _norm_date_iso(rec.get('VictimBirthDate')),
                })
    bzk = pd.DataFrame(rows)
    log.info(f"JSONL records loaded: {len(bzk):,}")

    # Allow first name to be missing on either side (OCR may have dropped it),
    # but when both sides have a value they must agree.
    def _first_ok(a, b): return a == b or a == '' or b == ''

    log.info("Merge pass: victim last name + birth date...")
    sub = bzk[bzk['vic_bd'].notna() & bzk['vic_last'].ne('')]
    m   = pd.merge(sub, meta_df, left_on=['vic_last', 'vic_bd'],
                   right_on=['meta_last', 'meta_bd'], how='inner')
    m   = m[m.apply(lambda r: _first_ok(r['vic_first'], r['meta_first']), axis=1)]

    index = {}
    for _, row in m.iterrows():
        fn = row['filename']
        if fn not in index:
            index[fn] = {'source': row['source'],
                         'Bestellsignatur': row['Bestellsignatur'],
                         'meta_sd_iso': row['meta_sd']}
    log.info(f"  Matches: {len(m):,}  →  {len(index):,} unique filenames")
    return index


def filename_index():
    def _stem_to_gt(stem: str) -> Optional[str]:
        m = _FNAME_PAT.match(stem)
        if not m: return None
        yyyy, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if mm == 0: return f'{yyyy:04d}-01-01'
        if dd == 0: return f'{yyyy:04d}-{mm:02d}-01'
        return f'{yyyy:04d}-{mm:02d}-{dd:02d}'

    _map_df = pd.read_csv(BZK_DATA_DIR / 'matched_offices_by_fullname_2.csv')
    _map_df = _map_df[~_map_df['Layout class'].isin(_IGNORE_CLASSES)]

    filename_index = {}
    skipped_ambiguous = 0
    for _, row in _map_df.iterrows():
        fname = row['Filename']
        lc    = row['Layout class']
        stem  = fname.rsplit('.', 1)[0]
        gt    = _stem_to_gt(stem)
        if gt is None: continue
        field = _LAYOUT_FILENAME_FIELD.get(lc)
        if field is None:          # ambiguous layout — filename unreliable
            skipped_ambiguous += 1
            continue
        filename_index[fname] = (gt, lc, field)
    return filename_index




def resolve_dates(rec: dict, merge_index: dict, fname_index: dict) -> dict:
    """
    Resolve all three date fields for one BZK record.

    Priority per field:
      1. metadata       — from merge_index 
      2. filename       — date encoded in filename 
      3. normalized     — normalize_date on the raw OCR value
      4. unparsed       — raw OCR value present but unparseable
      5. missing        — raw OCR value absent (OCR failure or blank on card)
      6. not_applicable — field structurally absent for this layout class
    """
    fname        = rec.get('filename', '')
    hit          = merge_index.get(fname)
    fi           = fname_index.get(fname)
    layout_class = fi[1] if fi else None
    ctx          = {'layout_class': layout_class}
    absent       = _LAYOUT_ABSENT_FIELDS.get(layout_class, set())
    result       = {}

    for field in _DATE_FIELDS:
        value = source = None
        uncertainty = 0

        # Priority 1 — metadata (VictimDeathDate only)
        if hit and field == 'VictimDeathDate':
            sd = hit.get('meta_sd_iso')
            if sd: value, source = sd, 'metadata'

        # Priority 2 — filename
        if value is None and fi is not None:
            gt_iso, _lc, fname_field = fi
            if fname_field == field:
                value, source = gt_iso, 'filename'

        # Priority 3 — normalization pipeline
        if value is None:
            raw = (rec.get(field) or '').strip()
            if raw and raw.lower() not in _SKIP_RAW_SET:
                norm, uncertainty = normalize_date(raw, col=field, ctx=ctx)
                value, source = (norm, 'normalized') if norm else ('', 'unparsed')
            elif field in absent:
                value, source = '', 'not_applicable'
            else:
                value, source = '', 'missing'

        result[field] = {'value': value or '', 'source': source, 'uncertainty': uncertainty}

        # Feed resolved VictimDeathDate into ctx for birth year century disambiguation
        if field == 'VictimDeathDate' and value:
            ctx['death_year'] = _extract_year(value)

    return result




def resolve_all_dates():
    log.info("=== resolve_all_dates start ===")
    log.info("Step 1/4: loading metadata...")
    _meta_df    = _load_metadata()

    log.info("Step 2/4: building merge index...")
    _merge_index = _build_merge_index(BZK_DATA_DIR, _meta_df)

    log.info("Step 3/4: building filename index...")
    _filename_index = filename_index()
    log.info(f"  Filename index: {len(_filename_index):,} entries")

    log.info(f"Step 4/4: resolving dates → {OUTPUT_PATH}")
    source_counts = {f: defaultdict(int) for f in _DATE_FIELDS}
    n_records = 0
    jsonl_files = sorted(BZK_DATA_DIR.glob('*.jsonl'))
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as out:
        for i, _jsonl in enumerate(jsonl_files, 1):
            log.info(f"  [{i}/{len(jsonl_files)}] {_jsonl.name}  (total so far: {n_records:,})")
            with open(_jsonl, encoding='utf-8') as fh:
                for line in fh:
                    rec = json.loads(line)
                    resolved = resolve_dates(rec, _merge_index, _filename_index)

                    out_rec = {'filename': rec.get('filename', '')}
                    for field in _DATE_FIELDS:
                        r = resolved[field]
                        out_rec[field]                  = r['value']
                        out_rec[field + '_source']      = r['source']
                        out_rec[field + '_uncertainty'] = r['uncertainty']
                        source_counts[field][r['source']] += 1

                    out.write(json.dumps(out_rec, ensure_ascii=False) + '\n')
                    n_records += 1

    log.info(f"Written {n_records:,} records → {OUTPUT_PATH}")

    # Source breakdown per field
    all_sources = ['metadata', 'filename', 'normalized', 'unparsed', 'missing', 'not_applicable']
    col_w = 16
    print(f"\n{'Field':<24} " + '  '.join(f'{s:>{col_w}}' for s in all_sources))
    print('-' * (24 + (col_w + 2) * len(all_sources)))
    for field in _DATE_FIELDS:
        total = sum(source_counts[field].values()) or 1
        row = f'{field:<24} '
        for s in all_sources:
            n = source_counts[field][s]
            row += f'  {f"{n:,} ({100*n/total:.0f}%)":>{col_w}}'
        print(row)
    log.info("=== resolve_all_dates done ===")

if __name__ == "__main__":
    resolve_all_dates()
