"""Deterministic English/Spanish rule extractor for the 12 RSNA knee targets.

Design (conservative by construction -- every doubtful case becomes `uncertain`, never `negative`):
  1. Sections: indication/history/technique/comparison text is ignored; it describes why the scan was
     ordered, not what it showed.
  2. Sentences -> hard clauses (but/however/while/...) -> pieces. A clause is split into pieces
     (at commas/and/or/with) only where BOTH sides name a target structure, so "tear of the
     posterior and anterior horns of the medial meniscus" stays whole while
     "ACL intact and medial meniscus tear" separates and one target's negation or lesion word cannot
     leak into the other's.
  3. Each piece is judged per target: negation (prefix/suffix, scoped to the piece), normal-structure
     phrases, hedging, historical/post-surgical context, severity qualifiers.
  4. Bare list items inherit negation from a neighbouring piece ("no tear of the ACL or MCL").
  5. Observations are aggregated per target: specific mentions override generic ones; positive+negative
     or any hedged mention together -> uncertain (reason recorded).

Unsupported languages are never guessed at (see report_label_language).

Known limits: lexical only, no parsing of long-range coreference; severity thresholds for
OA/effusion are annotation-dependent and are deliberately routed to `uncertain`. The extractor must be
validated against gold labels (report_label_eval) before its output is used for training.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Optional

from .report_label_language import detect_language, is_supported
from .report_labels import LabelState, ReportLabels, TargetLabel, unsupported_labels
from .rsna_knee_dataset import TARGET_COLUMNS

EXTRACTOR_VERSION = "rules-en-es-0.1.0"
EXTRACTOR_VERSION_V2 = "rules-en-es-0.2.0"  # opt-in: inline-heading split + weak findings subordinate to decisive ones
WEAK_REASONS = frozenset({"weak_finding", "generic_location", "negated_weak"})
METHOD = "rules"

# ----------------------------------------------------------------------------------------- text utils
_ABBREVIATIONS = ("vs.", "e.g.", "i.e.", "approx.", "cm.", "mm.", "dr.", "fig.", "ap.", "aprox.", "etc.", "no.")


def fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()


_HEADINGS = {
    "findings": "findings", "hallazgos": "findings", "report": "findings", "informe": "findings",
    "resultados": "findings", "resultado": "findings",
    "impression": "impression", "impressions": "impression", "conclusion": "impression",
    "conclusions": "impression", "impresion": "impression", "conclusiones": "impression",
    "opinion": "impression", "diagnostico": "impression", "resumen": "impression",
    "technique": "IGNORE", "tecnica": "IGNORE", "protocol": "IGNORE", "sequences": "IGNORE",
    "indication": "IGNORE", "indicacion": "IGNORE", "clinical indication": "IGNORE",
    "clinical history": "IGNORE", "history": "IGNORE", "clinical information": "IGNORE",
    "clinical": "IGNORE", "antecedentes": "IGNORE", "antecedentes clinicos": "IGNORE",
    "clinical data": "IGNORE", "reason for exam": "IGNORE", "reason for examination": "IGNORE",
    "comparison": "IGNORE", "comparacion": "IGNORE", "comparisons": "IGNORE", "exam type": "IGNORE",
    "exam": "IGNORE", "examination": "IGNORE", "exam date and time": "IGNORE", "estudio": "IGNORE",
    "procedure": "IGNORE", "motivo": "IGNORE", "motivo de consulta": "IGNORE",
}
_HEADING_LINE = re.compile(r"^\s*([a-z][a-z ]{2,30}?)\s*:\s*(.*)$")


_ACCENT_CLASSES = {"a": "[aáàâä]", "e": "[eéèêë]", "i": "[iíìîï]", "o": "[oóòôö]", "u": "[uúùûü]", "n": "[nñ]"}


def _accent_tolerant(word: str) -> str:
    return "".join(_ACCENT_CLASSES.get(ch, re.escape(ch)) for ch in word)


_INLINE_HEADING = re.compile(
    r"(?:(?<=[\s.;:!?])|(?<=^))(" + "|".join(_accent_tolerant(h).replace(" ", r"\s+")
                                              for h in sorted(_HEADINGS, key=len, reverse=True)) + r")\s*:",
    re.IGNORECASE)


def split_inline_headings(text: str) -> str:
    """Put every recognised section heading that sits in the middle of a line on a line of its own.

    Many reports are written as one paragraph ("Technique: ... Findings: ... Impression: ..."). Without this, a leading ignored
    heading (technique / indication) swallows the whole paragraph, findings included."""
    return "\n".join(_INLINE_HEADING.sub(lambda m: "\n" + m.group(0).lstrip(), line) for line in text.splitlines())


def split_sections(text: str, inline_headings: bool = False) -> list[tuple[str, str]]:
    """Return [(section_name, text)] with ignored sections dropped. Lines that continue a sentence are joined."""
    if inline_headings:
        text = split_inline_headings(text)
    sections, current, buffer = [], "body", []

    def flush():
        if buffer:
            sections.append((current, "\n".join(buffer)))
            buffer.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = _HEADING_LINE.match(fold(line))
        if match and match.group(1).strip() in _HEADINGS:
            flush()
            current = _HEADINGS[match.group(1).strip()]
            rest = line[line.index(":") + 1:].strip()
            if rest:
                buffer.append(rest)
            continue
        buffer.append(line)
    flush()
    return [(name, body) for name, body in sections if name != "IGNORE"]


def split_sentences(block: str) -> list[str]:
    protected = block
    for abbreviation in _ABBREVIATIONS:
        protected = re.sub(re.escape(abbreviation), abbreviation.replace(".", "\x00"), protected, flags=re.I)
    # a newline before a lowercase letter is a hard wrap inside a sentence
    protected = re.sub(r"\n(?=[a-zñáéíóú])", " ", protected)
    parts = re.split(r"(?<=[.;!?])\s+|\n", protected)
    return [p.replace("\x00", ".").strip() for p in parts if p.strip()]


_CLAUSE_BREAK = re.compile(
    r"\b(?:but|however|whereas|while|although|though|except(?: for)?|apart from|aside from|other than|"
    r"pero|sin embargo|aunque|excepto|salvo|mientras|a excepcion de)\b")
_PIECE_DELIM = re.compile(r",|\b(?:and|or|with|plus|as well as|y|o|e|con|asi como)\b")

# ----------------------------------------------------------------------------------------- cue lexicons
_NEG_PRE = re.compile(
    r"\b(?:no|not|without|negative for|absent|absence of|free of|nor|neither|sin|ausencia de|"
    r"no evidence of|no se (?:observa|observan|aprecia|aprecian|identifica|identifican|visualiza|evidencia|evidencian|"
    r"detecta|detectan|demuestra)\w*|no hay|no existe\w*|no presenta\w*|sin evidencia de|sin signos de)\b")
_NEG_SUF = re.compile(
    r"\b(?:(?:is|are|was|were|appears?|seems?)\s+)?(?:not (?:seen|identified|visuali[sz]ed|evident|present|demonstrated|"
    r"observed|appreciated)|absent|negative|ausente|ausentes)\b|\bno se (?:observa|aprecia|identifica)\w*")
_NORMAL = re.compile(
    r"\b(?:intact|intacto|intacta|intactos|intactas|normal|normales|unremarkable|preserved|conserved|conservad[oa]s?|"
    r"integr[oa]s?|within normal limits|dentro de limites normales|sin alteracion(?:es)?|sin hallazgos|"
    r"no abnormalit\w*|smooth|congruent|congruente|well[- ]maintained|maintained|mantenid[oa]s?)\b")
_NORMAL_V2 = re.compile(_NORMAL.pattern + r"|\b(?:continuous|continuo|continuos|continua|continuas)\b")
_HEDGE_EXTRA = r"\b(?:r/o|r\.o\.|rule[ -]outs?|a descartar|por descartar)(?!\w)"
_NOT_EXCLUDED = re.compile(
    r"\b(?:cannot|can not|can't|could not|unable to|difficult to)\s+(?:be\s+)?(?:fully\s+)?(?:exclude|excluded|rule out|ruled out)\w*|"
    r"\bnot (?:fully )?(?:excluded|ruled out)\b|\bno se (?:puede|pueden)? ?descartar\w*|\bno puede descartarse\b|\bno se descarta\b|"
    r"\bcabe descartar\b|\bno (?:se )?puede(?:n)? excluirse\b")
_HEDGE = re.compile(
    r"\b(?:possible|possibly|probable|probably|suspected|suspicious|suspect|may (?:represent|be|reflect|indicate)|might|"
    r"questionable|equivocal|versus|vs|suggest(?:s|ed|ive|ing)?|likely|favou?rs?|indeterminate|"
    r"posible|posibles|probables|sospech\w*|sugiere|sugerente|sugestiv\w*|podria|compatible con|dudos\w*|"
    r"indeterminad\w*|cuestionable|raises? (?:the )?possibility)\b")
_HEDGE_V2 = re.compile(_HEDGE.pattern + "|" + _HEDGE_EXTRA)
_HISTORICAL = re.compile(
    r"\b(?:post[- ]?operative|post[- ]?surgical|postquirurgic\w*|postoperatori\w*|status post|s/p|prior|previous\w*|"
    r"previa\w*|previo\w*|reconstruct\w*|reconstruid\w*|plastia|repaired|repair|reparad\w*|graft\w*|injerto\w*|"
    r"meniscectom\w*|sutur\w*|surgery|surgical|cirugia|quirurgic\w*|healed|cicatriz\w*|old|remote|sequela\w*|"
    r"secuela\w*|antigu\w*|consolidad\w*|fixation|hardware|orif|screws?|tornillos?|anchors?)\b")
_NOT_VISUALISED = re.compile(
    r"\bnot (?:well |clearly |adequately |optimally )?(?:visuali[sz]ed|evaluated|assessed|seen)\b|"
    r"\bno se (?:visualiza|valora|identifica|aprecia)\b|\bnon[- ]visuali[sz]\w*\b|\bobscured\b|\blimited (?:by|evaluation)\b")
_MILD = re.compile(
    r"\b(?:trace|minimal|minimum|tiny|scant|slight|physiologic\w*|fisiologic\w*|minim[oa]s?|escas[oa]s?|"
    r"ligera\w*|ligero\w*|early|incipient\w*|inicial\w*|grade (?:1|i|2|ii)\b|grade (?:one|two))\b")
_OA_MILD = re.compile(r"\b(?:mild|leve|slight|minimal|minimum|early|incipient\w*|inicial\w*|minim[oa]s?|ligera\w*|"
                      r"ligero\w*|grade (?:1|i|2|ii)\b)")
_STRONG = re.compile(
    r"\b(?:severe|marked|advanced|extensive|full[- ]thickness|espesor total|grave|severo|severa|avanzad[oa]|"
    r"grade (?:3|iii|4|iv)\b|bone[- ]on[- ]bone)\b")
_BURSA_PRE = re.compile(r"(?:burs\w*|sheath|vaina|baker|cyst|quiste)\W+(?:\w+\W+){0,3}$")


@dataclass(frozen=True)
class Spec:
    target: str
    kind: str  # "structure" (tear-type) or "presence" (the anchor itself is the finding)
    anchor: re.Pattern
    generic: Optional[re.Pattern] = None
    positive: Optional[re.Pattern] = None  # lesion terms (structure kinds)
    weak: Optional[re.Pattern] = None  # ambiguous lesion terms -> uncertain
    degenerative_context: Optional[re.Pattern] = None  # presence kinds that downgrade to uncertain
    normal_anchor: Optional[re.Pattern] = None  # phrases that can only ever support a negative


def _c(pattern: str) -> re.Pattern:
    return re.compile(pattern)


_LIG_POS = _c(r"\b(?:tear\w*|torn|rupture\w*|rotura\w*|ruptura\w*|disrupt\w*|discontinu\w*|avulsion\w*|transect\w*|avulsion|fibras rotas)\b")
_LIG_WEAK = _c(r"\b(?:sprain\w*|strain\w*|esguince\w*|distension|attenuat\w*|thicken\w*|edema|oedema|mucoid|laxity|lax|"
               r"stretch\w*|elongad\w*|lesion\w*|injur\w*|insufficien\w*|signal|senal)\b")
_MEN_POS = _c(r"\b(?:tear\w*|torn|rotura\w*|ruptura\w*|fissur\w*|fisura\w*|flap|bucket[- ]handle|asa de cubo|cleavage|"
              r"macerat\w*|macerad\w*|displaced fragment|fragmento desplazado)\b")
_MEN_WEAK = _c(r"\b(?:degenerat\w*|myxoid|mucoid|intrasubstance|intrameniscal\w*|signal|senal|extrusion|extruded|extrusi\w*|"
               r"amputaci\w*|truncat\w*|lesion\w*|injur\w*|meniscectom\w*|fraying|deshilach\w*)\b")

SPECS: dict[str, Spec] = {
    "ACL": Spec("ACL", "structure",
                _c(r"\b(?:anterior cruciate|acl|lca|cruzado anterior|ligamento cruzado anterior)\b"),
                _c(r"\b(?:cruciate ligaments?|cruciates?|ligamentos? cruzados?)\b"), _LIG_POS, _LIG_WEAK),
    "MCL": Spec("MCL", "structure",
                _c(r"\b(?:medial collateral|mcl|tibial collateral|colateral(?:es)? (?:medial|tibial)|"
                   r"ligamento colateral (?:medial|tibial|interno)|medial (?:and|&) lateral collateral|"
                   r"medial y lateral colateral)\b"),
                _c(r"(?<!lateral )(?<!fibular )(?<!lateral\) )\b(?:collateral ligaments?|collateral complex\w*|colaterales)\b"),
                _LIG_POS, _LIG_WEAK),
    "Medial Meniscus": Spec("Medial Meniscus", "structure",
                _c(r"\b(?:medial menisc\w*|menisc\w* medial\w*|menisco(?:s)? interno\w*|medial (?:and|&) lateral menisc\w*|"
                   r"menisc\w* medial y lateral)\b"),
                _c(r"\b(?:both menisc\w*|menisci|ambos meniscos|meniscos)\b"), _MEN_POS, _MEN_WEAK),
    "Lateral Meniscus": Spec("Lateral Meniscus", "structure",
                _c(r"\b(?:lateral menisc\w*|menisc\w* lateral\w*|menisco(?:s)? externo\w*|medial (?:and|&) lateral menisc\w*|"
                   r"menisc\w* medial y lateral)\b"),
                _c(r"\b(?:both menisc\w*|menisci|ambos meniscos|meniscos)\b"), _MEN_POS, _MEN_WEAK),
    "Effusion": Spec("Effusion", "presence",
                _c(r"\b(?:effusions?|derrames? (?:articular\w*|sinovial\w*)?|hydrarthrosis|hidrartros\w*|joint fluid|"
                   r"fluid in the joint|liquido (?:articular|intraarticular|sinovial))\b")),
    "Synovitis": Spec("Synovitis", "presence",
                _c(r"\b(?:synovitis|sinovitis|synovial (?:thickening|hypertrophy|proliferation|enhancement|inflammation|"
                   r"reaction|hyperplasia)|thickened synovium|hipertrofia sinovial|engrosamiento sinovial|"
                   r"proliferacion sinovial|inflamacion sinovial)\b")),
    "Baker's": Spec("Baker's", "presence",
                _c(r"\b(?:baker\W?s? cysts?|bakers? cysts?|quistes? de baker|popliteal (?:fossa )?cysts?|"
                   r"cysts? (?:in|within|of) the popliteal|quistes? poplite\w*|quiste sinovial poplite\w*|"
                   r"gastrocnemius[- ]semimembranosus bursa|bursa semimembranos\w*)\b")),
    "Contusion": Spec("Contusion", "presence",
                _c(r"\b(?:bone contusions?|bony contusions?|osseous contusions?|bone bruis\w*|contusion\w*|contusi\w+ osea\w*|"
                   r"(?:bone )?marrow (?:edema|oedema)|edema (?:de medula )?oseo|edema medular|osseous edema|bone edema|"
                   r"microtrabecular\w*|trabecular (?:injury|microfracture\w*))\b"),
                degenerative_context=_c(r"\b(?:subchondral|subcondral|degenerat\w*|osteoarth\w*|osteoartr\w*|reactive|reactivo\w*|"
                                        r"osteophyt\w*|stress|estres)\b"),
                normal_anchor=_c(r"\b(?:(?:bone )?marrow (?:signal|intensity)|senal (?:de la )?medula osea|medula osea|"
                                 r"alteraciones de senal de la medula)\b")),
    "Fracture": Spec("Fracture", "presence",
                _c(r"\b(?:fractur\w*|microfractur\w*)\b")),
}

# Osteoarthritis: three compartments share one vocabulary and are separated by the side words in the piece.
_OA_TERM = _c(
    r"\b(?:osteoarthr\w*|osteoartr\w*|artrosis|arthrosis|osteophyt\w*|osteofit\w*|chondromalac\w*|condromalac\w*|"
    r"chondrosis|condrosis|chondropath\w*|condropat\w*|cartilage (?:loss|thinning|defects?|fissur\w*|ulcerat\w*|wear|"
    r"erosions?|delaminat\w*)|(?:loss|thinning|defects?|fissur\w*|ulcer\w*|wear) (?:of|in) (?:the )?(?:articular )?cartilage|"
    r"chondral (?:defects?|lesions?|injur\w*|fissur\w*|loss|ulcer\w*|wear)|(?:ulcera|defecto|lesion|fisura)\w* condral\w*|"
    r"condrales?|perdida de cartilago|adelgazamiento (?:del )?cartilago|pinzamiento|joint space narrowing|"
    r"subchondral (?:cysts?|sclerosis|esclerosis|quistes?)|degenerative (?:changes?|joint disease|arthropath\w*|"
    r"arthritis|process\w*)|cambios degenerativos|enfermedad degenerativa|artropatia|geodes?)\b")
_OA_GENERIC_NEG_OK = _c(r"osteoarthr|osteoartr|artrosis|arthrosis|degenerative|degenerativ|chondromalac|condromalac|cartilage|"
                        r"cartilago|chondral|condral|chondrosis|condrosis")
_CART_NOUN = _c(r"\b(?:cartilage\w*|cartilago\w*|chondral|condral\w*|joint space|espacio articular)\b")
_SIDE_MED = _c(r"\b(?:medial|internal|interno|interna)\s+(?:femorotibial\s+|tibiofemoral\s+|femoral\s+|tibial\s+)?"
               r"(?:compartment|condyle|plateau|joint space|joint|cartilage|compartimento|condilo|platillo|espacio|cartilago)\b|"
               r"\bcompartimento (?:femorotibial )?(?:medial|interno)\b|\bfemorotibial medial\b|\bmedial femorotibial\b")
_SIDE_LAT = _c(r"\b(?:lateral|external|externo|externa)\s+(?:femorotibial\s+|tibiofemoral\s+|femoral\s+|tibial\s+)?"
               r"(?:compartment|condyle|plateau|joint space|joint|cartilage|compartimento|condilo|platillo|espacio|cartilago)\b|"
               r"\bcompartimento (?:femorotibial )?(?:lateral|externo)\b|\bfemorotibial lateral\b|\blateral femorotibial\b")
_SIDE_BOTH = _c(r"\b(?:femorotibial|tibiofemoral) (?:compartments|joints)\b|\bcompartimentos femorotibiales\b|"
                r"\bmedial (?:and|&) lateral (?:femorotibial )?compartments?\b|\bboth (?:femorotibial )?compartments\b|"
                r"\bcompartimentos medial y lateral\b|\bfemorotibial\b|\btibiofemoral\b")
_SIDE_PF = _c(r"\b(?:patellofemoral|femoropatell?ar|femoropatelar|patellar (?:cartilage|chondr\w*|facet|surface|apex|ridge)|"
              r"patella|trochlea\w*|troclea\w*|rotulian\w*|patelar|retropatell?ar\w*|pf joint)\b")
_STRUCTURE_CONTEXT = _c(r"\b(?:menisc\w*|ligament\w*|tendon\w*|retinacul\w*|meniscus)\b")

_SENTENCE_STARTS_NEG = re.compile(r"^\W*(?:no|not|without|sin|nor|neither)\b")


# ----------------------------------------------------------------------------------------- observations
@dataclass
class Observation:
    target: str
    state: LabelState
    specific: bool
    section: str
    evidence: str
    reason: Optional[str] = None


def _negated(piece: str, spans: Iterable[tuple[int, int]], *, normal_ok: bool, v2: bool = False) -> bool:
    spans = list(spans)
    if not spans:
        return False
    first, last = min(s for s, _ in spans), max(e for _, e in spans)
    protected = _NOT_EXCLUDED.sub(" ", piece)
    for cue in _NEG_PRE.finditer(protected):
        if cue.start() < last:
            return True
    if _NEG_SUF.search(protected, first):
        return True
    return bool(normal_ok and (_NORMAL_V2 if v2 else _NORMAL).search(protected))


def _spans(pattern: Optional[re.Pattern], piece: str) -> list[tuple[int, int]]:
    return [m.span() for m in pattern.finditer(piece)] if pattern is not None else []


def _modify(state: LabelState, piece: str, reason: Optional[str], v2: bool = False) -> tuple[LabelState, Optional[str]]:
    """Hedging and historical context downgrade a decided state to uncertain; never the other way round."""
    if state in (LabelState.POSITIVE, LabelState.NEGATIVE):
        if _NOT_EXCLUDED.search(piece) or (_HEDGE_V2 if v2 else _HEDGE).search(_NOT_EXCLUDED.sub(" ", piece)):
            return LabelState.UNCERTAIN, "hedged"
        if _HISTORICAL.search(piece):
            return LabelState.UNCERTAIN, "historical"
    return state, reason


def _positive_precedes_negation(piece: str, first_positive: int) -> bool:
    """True when the piece has a negation cue but none of it comes before the first explicit lesion term (prefix or suffix)."""
    protected = _NOT_EXCLUDED.sub(" ", piece)
    cues = [m.start() for m in _NEG_PRE.finditer(protected)]
    if _NEG_SUF.search(protected, first_positive):
        return False
    return bool(cues) and min(cues) > first_positive


def _judge(piece: str, spec: Spec, v2: bool = False) -> Optional[tuple[LabelState, bool, Optional[str]]]:
    """Judge one piece for one target. Returns (state, specific, reason) or None when the piece says nothing."""
    specific_hits = _spans(spec.anchor, piece)
    generic_hits = _spans(spec.generic, piece) if not specific_hits else []
    anchors = specific_hits or generic_hits
    normal_hits = _spans(spec.normal_anchor, piece) if not anchors else []
    if not anchors and not normal_hits:
        return None
    specific = bool(specific_hits) or (not anchors)

    if spec.kind == "presence":
        if not anchors:  # normal_anchor only: can support a negative, never a positive
            if _negated(piece, normal_hits, normal_ok=True, v2=v2):
                return LabelState.NEGATIVE, specific, "normal_phrase"
            return None
        valid = [(s, e) for s, e in anchors if not _BURSA_PRE.search(piece[:s])] if spec.target == "Effusion" else anchors
        if not valid:
            return None
        if _negated(piece, valid, normal_ok=False, v2=v2):
            state, reason = _modify(LabelState.NEGATIVE, piece, None, v2)
            return state, specific, reason
        state, reason = LabelState.POSITIVE, None
        if spec.degenerative_context is not None and spec.degenerative_context.search(piece):
            state, reason = LabelState.UNCERTAIN, "degenerative_context"
        elif _MILD.search(piece) and spec.target in ("Effusion", "Synovitis", "Baker's"):
            state, reason = LabelState.UNCERTAIN, "trace_or_minimal"
        else:
            state, reason = _modify(state, piece, reason, v2)
        return state, specific, reason

    pos, weak = _spans(spec.positive, piece), _spans(spec.weak, piece)
    lesion_spans = pos + weak
    if _NOT_VISUALISED.search(piece) and not pos:
        return LabelState.UNCERTAIN, specific, "not_visualized"
    if v2 and pos and _positive_precedes_negation(piece, min(a for a, _ in pos)):
        # "tear of the posterior horn ... without extrusion": the cue negates the later weak term, not the tear before it
        state, reason = _modify(LabelState.POSITIVE, piece, None, v2)
    elif lesion_spans and _negated(piece, lesion_spans, normal_ok=True, v2=v2):
        # v2: a negation that only reaches a weak term ("not all in a straight line ... injury") is itself a weak statement
        state, reason = _modify(LabelState.NEGATIVE, piece, "negated_weak" if v2 and not pos else "negated_lesion", v2)
    elif pos:
        state, reason = _modify(LabelState.POSITIVE, piece, None, v2)
    elif weak:
        state, reason = LabelState.UNCERTAIN, ("historical" if v2 and _HISTORICAL.search(piece) else "weak_finding")
    elif (_NORMAL_V2 if v2 else _NORMAL).search(piece) or re.search(r"\bno abnormal\w*", piece):
        state, reason = _modify(LabelState.NEGATIVE, piece, "normal_structure", v2)
    elif _HISTORICAL.search(piece):
        state, reason = LabelState.UNCERTAIN, "historical"  # e.g. "status post ACL reconstruction"
    else:
        return None  # bare mention: may inherit from a neighbouring list item
    if not specific_hits and state is LabelState.POSITIVE:
        state, reason = LabelState.UNCERTAIN, "generic_location"
    return state, specific, reason


def _judge_oa(piece: str, inherited_sides: frozenset, v2: bool = False) -> list[tuple[str, LabelState, bool, Optional[str]]]:
    terms = _spans(_OA_TERM, piece)
    cart = _spans(_CART_NOUN, piece)
    if not terms and not cart:
        return []
    sides = set()
    if _SIDE_MED.search(piece):
        sides.add("Medial OA")
    if _SIDE_LAT.search(piece):
        sides.add("Lateral OA")
    if _SIDE_BOTH.search(piece):
        sides.update(("Medial OA", "Lateral OA"))
    if _SIDE_PF.search(piece):
        sides.add("PF OA")
    carried = False
    if not sides and inherited_sides:
        sides, carried = set(inherited_sides), True
    negated = _negated(piece, terms or cart, normal_ok=True, v2=v2)
    if terms and negated:
        state, reason = LabelState.NEGATIVE, "negated_lesion"
    elif terms:
        state, reason = LabelState.POSITIVE, None
        if _OA_MILD.search(piece) and not _STRONG.search(piece):
            state, reason = LabelState.UNCERTAIN, "mild_or_early"
    elif negated:  # cartilage noun with a normal / negation phrase and no lesion term
        state, reason = LabelState.NEGATIVE, "normal_cartilage"
    else:
        return []
    state, reason = _modify(state, piece, reason, v2)
    out = []
    if sides:
        for target in sorted(sides):
            out.append((target, state, not carried, reason if not carried else (reason or "inherited_side")))
    else:  # location unknown: only an explicit "no OA / normal cartilage" statement is applied to all three
        if state is LabelState.NEGATIVE and _OA_GENERIC_NEG_OK.search(piece):
            out = [(t, state, False, "generic_location") for t in ("Medial OA", "Lateral OA", "PF OA")]
        elif state is not LabelState.NEGATIVE and not _STRUCTURE_CONTEXT.search(piece):
            out = [(t, LabelState.UNCERTAIN, False, "generic_location") for t in ("Medial OA", "Lateral OA", "PF OA")]
    return out


def _has_anchor(text: str) -> bool:
    for spec in SPECS.values():
        if spec.anchor.search(text) or (spec.generic and spec.generic.search(text)) or \
                (spec.normal_anchor and spec.normal_anchor.search(text)):
            return True
    # OA lesion/cartilage words are deliberately not anchors: "no focal chondrosis or chondral injury" is one thought
    return bool(_SIDE_MED.search(text) or _SIDE_LAT.search(text) or _SIDE_PF.search(text))


def split_pieces(clause: str) -> list[str]:
    """Split at list delimiters only when both neighbouring segments name a target structure."""
    boundaries = [(m.start(), m.end()) for m in _PIECE_DELIM.finditer(clause)]
    pieces, start = [], 0
    for index, (b_start, b_end) in enumerate(boundaries):
        right_end = boundaries[index + 1][0] if index + 1 < len(boundaries) else len(clause)
        left, right = clause[start:b_start], clause[b_end:right_end]
        if _has_anchor(left) and _has_anchor(right):
            pieces.append(clause[start:b_start])
            start = b_start
    pieces.append(clause[start:])
    return [p for p in pieces if p.strip()]


def _opens_with_negation(piece: str) -> bool:
    cue = _NEG_PRE.search(piece)
    anchor = spec_any_anchor(piece)
    return bool(cue and (anchor is None or cue.start() < anchor.start()))


_FILLER = re.compile(r"\b(?:the|a|an|of|any|or|and|nor|y|o|e|el|la|los|las|de|del|evidence|signs?|signos?|evidencia|"
                     r"significant|focal|definite|obvious|other|otros?)\b")


def _is_bare_item(piece: str, spec: Spec) -> bool:
    """True when a piece is just the target's name (plus list words), e.g. "or fracture"."""
    rest = spec.anchor.sub(" ", piece)
    rest = _FILLER.sub(" ", rest)
    return not re.search(r"[a-z]", rest)


def _inherit(pieces: list[str], judged: list[dict], spec: Spec, index: int):
    """A bare list item takes its state from a neighbour: negation from the previous piece when that piece opens
    with a negation cue ("no tear of the ACL or MCL"); otherwise from the next piece ("ACL and medial meniscus tears")."""
    neighbours = []
    if index > 0:
        neighbours.append((index - 1, _opens_with_negation(pieces[index - 1])))
    if index + 1 < len(pieces):
        neighbours.append((index + 1, False))
    for j, _ in sorted(neighbours, key=lambda item: not item[1]):
        for other_target, (state, _, _) in judged[j].items():
            same_family = SPECS[other_target].kind == spec.kind == "structure"
            if state is LabelState.NEGATIVE or (same_family and state in (LabelState.POSITIVE, LabelState.UNCERTAIN)):
                return state, False, "list_inherited"
    return None


def spec_any_anchor(text: str):
    best = None
    for spec in SPECS.values():
        for pattern in (spec.anchor, spec.generic):
            if pattern is not None:
                m = pattern.search(text)
                if m and (best is None or m.start() < best.start()):
                    best = m
    return best


class RuleBasedExtractor:
    """English/Spanish deterministic extractor. Thread-safe and stateless."""

    method = METHOD

    def __init__(self, profile: str = "0.1.0"):
        """profile "0.1.0" reproduces the original behaviour exactly (the Phase 5 manifest depends on it);
        profile "0.2.0" enables the two audited fixes and is versioned separately."""
        if profile not in ("0.1.0", "0.2.0"):
            raise ValueError("profile must be '0.1.0' or '0.2.0'")
        self.profile = profile
        self.version = EXTRACTOR_VERSION if profile == "0.1.0" else EXTRACTOR_VERSION_V2
        self._v2 = profile == "0.2.0"

    @property
    def source_sha256(self) -> str:
        """Hash of the rule and language sources, recorded in manifests as extractor provenance."""
        import hashlib
        from pathlib import Path
        here = Path(__file__).resolve().parent
        digest = hashlib.sha256()
        for name in ("report_label_rules.py", "report_label_language.py"):
            digest.update((here / name).read_bytes())
        return digest.hexdigest()

    def extract(self, report_text: str, study_uid: Optional[str] = None) -> ReportLabels:
        language = detect_language(report_text or "")
        if not is_supported(language):
            return unsupported_labels(language, self.method, self.version, study_uid)
        observations: list[Observation] = []
        for section, block in split_sections(report_text, inline_headings=self._v2):
            previous_sides: frozenset = frozenset()
            for sentence in split_sentences(block):
                folded = fold(sentence)
                sides_here = set()
                for clause in _CLAUSE_BREAK.split(folded):
                    if not clause or not clause.strip():
                        continue
                    pieces = split_pieces(clause)
                    judged = [{t: r for t, spec in SPECS.items() if (r := _judge(p, spec, self._v2)) is not None} for p in pieces]
                    for i, piece in enumerate(pieces):
                        for target, spec in SPECS.items():
                            result = judged[i].get(target)
                            if result is not None and spec.kind == "presence" and result[0] is LabelState.POSITIVE                                     and i > 0 and _is_bare_item(piece, spec) and _opens_with_negation(pieces[i - 1]):
                                result = (LabelState.NEGATIVE, True, "list_inherited")
                            if result is None and (spec.anchor.search(piece) or (spec.generic and spec.generic.search(piece))):
                                result = _inherit(pieces, judged, spec, i)
                            if result is not None:
                                state, specific, reason = result
                                observations.append(Observation(target, state, specific, section, sentence, reason))
                        for target, state, specific, reason in _judge_oa(piece, previous_sides, self._v2):
                            observations.append(Observation(target, state, specific, section, sentence, reason))
                        if _SIDE_MED.search(piece):
                            sides_here.add("Medial OA")
                        if _SIDE_LAT.search(piece):
                            sides_here.add("Lateral OA")
                        if _SIDE_PF.search(piece):
                            sides_here.add("PF OA")
                previous_sides = frozenset(sides_here)
        return self._aggregate(observations, language, study_uid)

    def _aggregate(self, observations: list[Observation], language: str, study_uid: Optional[str]) -> ReportLabels:
        labels = []
        for target in TARGET_COLUMNS:
            mine = [o for o in observations if o.target == target]
            specific = [o for o in mine if o.specific]
            pool = specific or mine
            if self._v2 and any(o.state in (LabelState.POSITIVE, LabelState.NEGATIVE) for o in pool):
                # a weak finding (degeneration, extrusion, "grade 1 injury", location-free mention) cannot veto an explicit
                # positive or negative statement about the same target; hedges and history still can
                decisive = [o for o in pool if o.state in (LabelState.POSITIVE, LabelState.NEGATIVE) and o.reason not in WEAK_REASONS]
                if decisive:  # weak statements are dropped; hedged / historical uncertain observations stay and can still veto
                    pool = [o for o in pool if o.reason not in WEAK_REASONS]
            states = {o.state for o in pool}
            if not pool:
                labels.append(TargetLabel(target, LabelState.NOT_MENTIONED, language, self.method, self.version))
                continue
            if len(states) == 1:
                state, reason = next(iter(states)), None
            elif states == {LabelState.POSITIVE, LabelState.NEGATIVE}:
                state, reason = LabelState.UNCERTAIN, "conflicting_mentions"
            else:
                state, reason = LabelState.UNCERTAIN, "mixed_certainty"
            chosen = sorted((o for o in pool if o.state is state or state is LabelState.UNCERTAIN),
                            key=lambda o: (o.section != "impression",))[0]
            labels.append(TargetLabel(target, state, language, self.method, self.version,
                                      evidence=chosen.evidence, section=chosen.section,
                                      reason=reason or chosen.reason))
        return ReportLabels(study_uid, language, True, self.method, self.version, tuple(labels))
