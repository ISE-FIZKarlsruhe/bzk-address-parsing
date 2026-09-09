import re
from typing import Optional, TypedDict, Literal, NamedTuple, NotRequired
import textwrap
type EntityType = Literal['City', 'Country', 'State', 'District', 'Neighborhood', 'Uncertain', 'AboveCity']

class SimplifiedEntity(NamedTuple):
    entity_type : EntityType
    geonames_id : str

class Match(TypedDict):
    text : str
    alternative_wordings : NotRequired[list[str]]
    start : int
    end : int
    geonames_id : NotRequired[str]
    special_regex : NotRequired[str]

class AddressMatches(TypedDict):
    matches : dict[str, Match]
    global_regex : NotRequired[re.Pattern]


# (?:Rum|Ung|Norw|Arg|Isr|Engl|Po|Pol|Belg|Ukr|Bulg|Austr|Boliv|Bol)\.
COMMON_NAMES = [
    (r'R[uü]m\.?', SimplifiedEntity(entity_type='Country', geonames_id='798549')),
    (r'Bulg\.?', SimplifiedEntity(entity_type='Country', geonames_id='732800')),
    (r'Isr\.?', SimplifiedEntity(entity_type='Country', geonames_id='294640')),
    (r'Engl\.?', SimplifiedEntity(entity_type='Country', geonames_id='6269131')),
    (r'Pol?\.?', SimplifiedEntity(entity_type='Country', geonames_id='798544')),
    (r'Belg?\.?', SimplifiedEntity(entity_type='Country', geonames_id='2802361')),
    (r'Ukr?\.?', SimplifiedEntity(entity_type='Country', geonames_id='690791')),
    (r'Ung\.?', SimplifiedEntity(entity_type='Country', geonames_id='719819')),
    (r'Austr\.?', SimplifiedEntity(entity_type='Country', geonames_id='2077456')),
    (r'CSR|ČSSR', SimplifiedEntity(entity_type='Country', geonames_id='8505031')),
    (r'Frankfurt\W*(am?)?\W*M(a(in?)?)?', SimplifiedEntity(entity_type='City', geonames_id='2925533')),
    (r'Ffm\.?', SimplifiedEntity(entity_type='City', geonames_id='2925533')),
    (r'U\.?S\.?A\.?', SimplifiedEntity(entity_type='Country', geonames_id='6252001')),
    (r'New\W*York(\W*City)?', SimplifiedEntity(entity_type='City', geonames_id='5128581')),
    (r'N\.?Y\.?', [SimplifiedEntity(entity_type='City', geonames_id='5128581'), SimplifiedEntity(entity_type='State', geonames_id='5128638')]),
]

compiled_common_names : list[tuple[re.Pattern, SimplifiedEntity | list[SimplifiedEntity]]] = []
for pattern, match in COMMON_NAMES:
    compiled_common_names.append((re.compile(r'(?:\W|^)(?P<Match>' + pattern + r')(?:\W|$)', re.IGNORECASE), match))

DISTRICT_KEYWORD_PATTERN = r'(kreis|krs?\.|(reg\.\s*)?bez\.)\s*'
STREET_PATTERN = (
    r'[^\d\W]+'                 
    r'(?:'
        r'(?:str(?:aße|\.)?)'                
        r'|(?:gasse|Gasse)'
        r'|(?:weg|Weg)'
        r'|(?:allee|Allee)'
        r'|(?:platz|Platz)'
        r'|(?:ring|Ring)'
        r'|(?:damm|Damm)'
        r'|(?:chaussee|Chaussee)'
        r'|(?:ufer|Ufer)'
        r'|(?:steig|Steig)'
    r')'
)

# regex credits to https://stackoverflow.com/questions/267399/how-do-you-match-only-valid-roman-numerals-with-a-regular-expression
# Original pattern: ^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$
# But we do not need the hundreds place.
ROMAN_NUMERAL_PATTERN = r'(?i:(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))'
# matches "digit sequence followed by a single letter or a roman numeral"
HOUSE_NR_PATTERN = fr'\d+\W*(([a-zA-Z]|{ROMAN_NUMERAL_PATTERN})(?:\W+|$))?'

# Regions proposed by CLAUDE AI
REGIONS = r"""(?:
    [^\d\W_](\.\b|\b)|
    main|rhein|neckar|elbe|oder|saale|spree|ruhr|weser|
    mosel|lahn|isar|inn|donau|havel|fulda|werra|nahe|
    regnitz|pegnitz|enz|kocher|jagst|tauber|wupper|
    sieg|ahr|nidda|leine|aller|ems|lippe|erft|
    schwarzwasser|
    oberlausitz|niederlausitz|lausitz|
    bergisches?\s*land|
    bergstra(?:ß|ss)e|
    breisgau|pfalz|holstein|allgäu|taunus|
    spreewald|fläming|vogtland|odenwald|spessart|
    hunsrück|eifel|westerwald|uckermark|altmark|
    prignitz|harz|
    bodensee|chiemsee|ammersee|starnberger\s*see|
    tegernsee
)"""

# Built with Claude support
CITY_NEARBY_FEATURE_GERMAN_REGEX_TEMPLATE = re.compile(
    r"""(?ix)
    \b
    (?P<City>
        [^\d\W_][^\d\W_]+
        \s*
        (?:
            # Frankfurt/Main, Frankfurt/M.
            /\s*""" + REGIONS + r"""
            |
            # Frankfurt (Oder), Wetter (Ruhr)
            \(\s*""" + REGIONS + r"""\s*\)
            |
            (?:
                a\.?\s*m\.?         # am, a.M., a. M.
              | a\.?\s*d\.?         # a.d.
              | an\s+der
              | an\s+dem
              | i\.?\s*m\.?         # im, i.M.
              | i\.?\s*d\.?         # i.d.
              | in\s+der
              | in\s+dem
              | ob\s+der            # Rothenburg ob der Tauber
              | b\.?                # b., bei
              | bei
            )
            \s*
            """ + REGIONS + r"""
        )
    )
    """
)


# NOTE: Disabled, brings false positives (churches, private propeties, etc.)
# SAINT_CITY_REGEX = re.compile(r"(?:\W|^)(?P<City>(St?\.|Saint|San(ta)?|S[ãa]o)\s+[^\d\W]+)(?:\W|$)")
# LOS_CITY_REGEX = re.compile(r"(?:\W|^)(?P<City>L[ao]s?\s+[^\d\W]+)(?:\W|$)")
# NEW_CITY_REGEX = re.compile(r"(?:\W|^)(?P<City>New\s+[^\d\W]+)(?:\W|$)")

# TODO use? It does not necessarily provide something useful
# 657 E 7th Street
CARDINAL_NUMBER_STREET_PATTERN = re.compile(r"(?P<HouseNumber>\d{1,4})\s+(?i:(?P<StreetName>([NSWE]|North|South|East|West)\s+\d{1,3}(st|nd|rd|th)?\s+(St\.|Street|Ave\.?|Avenue|Blvd\.?|Boulevard|Road|Rd\.?|Lane|Ln\.?|Drive|Dr\.?|Way)))")



CITY_KREIS_DIRSTRICT_REGEX = re.compile(fr"^(?P<City>[^\d\W]+)\W*{DISTRICT_KEYWORD_PATTERN}(?P<District>[^\d\W]+)?$", re.IGNORECASE)

CITY_SOMETHING_REGEX = re.compile(r"^(?P<City>[^\d\W]+)(\s*[/,]\(?\s*(?P<AboveCity>[^\d\W]+))?\s*\)?$")

CONCETRATION_CAMP_PATTERN = re.compile(r"^K\.?\s*Z\.?\s*(?P<Compound>[\w\s]+)$", re.IGNORECASE)
SUFFIX_CONCETRATION_CAMP_PATTERN = re.compile(r"^(?P<Compound>[\w\s]+)\s*K\.?\s*Z\.?$", re.IGNORECASE)

STREET_NUMBER_REGEX = re.compile(fr"^(?P<StreetName>{STREET_PATTERN})\s*(?P<HouseNumber>{HOUSE_NR_PATTERN})$")
CITY_STREET_NUMBER_REGEX = re.compile(fr"^(?P<City>[^\d\W]+),\s*(?P<StreetName>{STREET_PATTERN})\s*(?P<HouseNumber>{HOUSE_NR_PATTERN})$")
STREET_NUMBER_CITY_REGEX = re.compile(fr"^(?P<StreetName>{STREET_PATTERN})\s*(?P<HouseNumber>{HOUSE_NR_PATTERN}),\s*(?P<City>[^\d\W]+)$")

STREET_NUMBER_POSTAL_CITY_REGEX = re.compile(fr"^(?P<StreetName>{STREET_PATTERN})\s*(?P<HouseNumber>{HOUSE_NR_PATTERN}),\s*(?P<PostalCode>\d{{4,5}})\s*(?P<City>[^\d\W]+)$")

DEPORTATION_PATTERN = re.compile(r"(?P<Deportation>^(?:(in|der)\s+)*Deport(ation|iert))$", re.IGNORECASE)
MISSING_PERSON_PATTERN = re.compile(r"(?P<Missing>^(Verschollen|Verschwunden)$)", re.IGNORECASE)
UNKNOWN_PATTERN = re.compile(r"(?P<Unknown>^\W*$|^\s*unbekannt\s*$)", re.IGNORECASE)

# NOTE: Disabled, brings false positives
PARTIAL_REGEX_LIST = [
    CITY_NEARBY_FEATURE_GERMAN_REGEX_TEMPLATE
]

GLOBAL_REGEX_LIST = [
    CONCETRATION_CAMP_PATTERN,
    SUFFIX_CONCETRATION_CAMP_PATTERN,
    CITY_SOMETHING_REGEX,
    CITY_KREIS_DIRSTRICT_REGEX,
    STREET_NUMBER_REGEX,
    CITY_STREET_NUMBER_REGEX,
    STREET_NUMBER_CITY_REGEX,
    DEPORTATION_PATTERN,
    MISSING_PERSON_PATTERN,
    UNKNOWN_PATTERN
]

def get_words_left(address: str, still_unparsed: list[bool]) -> list[str]:
    words = []
    current_word = []
    in_word = False
    in_number = False
    def _flush_current_word():
        nonlocal current_word, words, in_word, in_number
        in_word = False
        in_number = False
        if current_word:
            words.append(''.join(current_word))
            current_word.clear()
    
    for i in range(len(address)):
        if not still_unparsed[i]:
            _flush_current_word()
        elif address[i].isalpha() and not in_word:
            _flush_current_word()
            in_word = True
        elif address[i].isdigit() and not in_number:
            _flush_current_word()
            in_number = True
        else:
            _flush_current_word()
        if in_word or in_number:
            current_word.append(address[i])
    return words

def regex_parse(address : str) -> dict[str, Match|str]:
    """
    Parse an address string and return a dictionary with the parsed components.

    Args:
        address (str): The address string to parse.

    Returns:
        dict[str, Match|str]: A dictionary with the parsed components.
    """
    if address is None or address == "" or not isinstance(address, str):
        return {}
    results = {}
    status = "unparsed"

    for name_pattern, match_candidates in compiled_common_names:
        pattern_match = name_pattern.search(address)
        if pattern_match:
            if isinstance(match_candidates, list):
                for candidate in match_candidates:
                    direct_match = candidate
                    if candidate.entity_type not in results:
                        break
            else:
                direct_match = match_candidates
            results[direct_match.entity_type] = Match(
                text=pattern_match.group("Match"),
                geonames_id=direct_match.geonames_id,
                start=pattern_match.start("Match"),
                end=pattern_match.end("Match"),
                special_regex=name_pattern.pattern.replace('\n', ' ')
            )
            status = "partially_parsed"

    for partial_regex in PARTIAL_REGEX_LIST:
        pattern_match = partial_regex.search(address)
        if pattern_match:
            for group_name, group_value in pattern_match.groupdict().items():
                if group_value is None:
                    continue
                already_matched = results.get(group_name)
                start = pattern_match.start(group_name)
                end = pattern_match.end(group_name)
                if already_matched is not None:
                    continue
                results[group_name] = Match(
                    text=address[pattern_match.start(group_name):pattern_match.end(group_name)],
                    start=pattern_match.start(group_name),
                    end=pattern_match.end(group_name),
                    special_regex=partial_regex.pattern.replace('\n', ' ')
                )
                status = "partially_parsed"


    still_unparsed_chars = list(address)
    still_unparsed = [True] * len(address)
    
    for result in results.values():
        for i in range(result['start'], result['end']):
            # replace non word characters in the still_unparsed string 
            # allowing other patterns to match it now
            still_unparsed[i] = False
            if not still_unparsed_chars[i].isalpha():
                still_unparsed_chars[i] = '_'

    words_left = get_words_left(address, still_unparsed)
    if len(words_left) == 0:
        results["status"] = "fully_parsed"
        return results
    elif len(words_left) == 1 and 'City' not in results:
        results['City'] = Match(
            text=words_left[0],
            start=address.find(words_left[0]),
            end=address.find(words_left[0]) + len(words_left[0])
        )
        results["status"] = "fully_parsed"
        return results
    
    enhanced_str = ''.join(still_unparsed_chars)
    if not any(still_unparsed): # TODO seems redundant with len(words_left) == 0
        results["status"] = "fully_parsed"
        return results
    for regex in GLOBAL_REGEX_LIST:
        pattern_match = regex.fullmatch(enhanced_str)
        if pattern_match:
            for group_name, group_value in pattern_match.groupdict().items():
                already_matched = results.get(group_name)
                start = pattern_match.start(group_name)
                end = pattern_match.end(group_name)
                if already_matched is not None:
                    continue
                if group_value is not None:
                    results[group_name] = Match(
                        text=address[start:end],
                        start=start,
                        end=end
                    )
            results['global_regex'] = regex.pattern.replace('\n', ' ')
            results["status"] = "fully_parsed"
            return results
    
    results["status"] = status
    return results