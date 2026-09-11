"""A regex + lexicon baseline for PII span detection.

The point of a baseline is to find out whether the adapter earns its place, so this is
written to be *good*, not to lose. A strawman baseline is the same mistake as a confounded
dataset: it produces a comfortable number that means nothing (STATUS.md D21).

What it can and cannot reach, from the golden set's label distribution:

  reachable by pattern   EMAIL, CREDITCARDNUMBER, TELEPHONENUM, SOCIALNUM, DATE, ZIPCODE,
                         PASSPORTNUM, DRIVERLICENSENUM, IDCARDNUM, TAXNUM   (~46% of spans)
  reachable by lexicon   TITLE, GENDER, SEX                                 (~12% of spans)
  needs NER              GIVENNAME, SURNAME, CITY, STREET                   (~33% of spans)
  genuinely ambiguous    AGE, BUILDINGNUM — bare 1-3 digit numbers that regex cannot
                         separate from each other or from ordinary quantities (~9%)

Reachable categories total ~58% of spans, so that is the structural ceiling on recall.
**Measured recall is 0.307** — barely half of it. The shortfall is not laziness in the
patterns: within the numeric types, ZIPCODE, AGE, BUILDINGNUM, TELEPHONENUM, SOCIALNUM,
TAXNUM and the various ID numbers are mutually ambiguous as character sequences, so a
pattern precise enough to avoid false positives on one of them misses most of another.
ZIPCODE lands at F1 0.166 for exactly this reason; EMAIL, which has a shape nothing else
shares, lands at 0.995.

That is the honest comparison for the adapter: it says how much of PII detection is
*pattern matching* (a little — one label) and how much needs a model that reads context.

`predict_hybrid` adds spaCy NER on top, which is what raises the name-like labels off zero
and makes this a serious baseline rather than a regex demo. Two mismatches have to be
handled: spaCy emits one `PERSON` span per name while the gold splits TITLE / GIVENNAME /
SURNAME, and spaCy sometimes tags an email address as a PERSON. Running the patterns first
and letting them claim characters fixes the second; the first needs `PERSON` decomposed by
token position, which is approximate — gold `GIVENNAME` is occasionally two words
("Vera-Maria Kélia"), so some boundaries will be wrong and strict F1 pays for it.
"""

from __future__ import annotations

import re

from adapterops.eval.spans import Span

TITLES = {"mr", "mrs", "ms", "miss", "dr", "prof", "sir", "madam", "master", "mstr",
          "mx", "lord", "lady", "rev", "capt", "major", "col", "sgt"}
GENDERS = {"male", "female", "non-binary", "nonbinary", "transgender", "man", "woman"}
SEXES = {"m", "f"}

# Order matters: the first pattern to claim a character span wins, so the most specific
# and least ambiguous patterns come first.
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")),
    ("CREDITCARDNUMBER", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("DATE", re.compile(
        r"\b(?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}"
        r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"
        r"|\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b")),
    ("SOCIALNUM", re.compile(r"\b\d{3}[.\- ]\d{3}[.\- ]\d{4}\b")),
    ("TELEPHONENUM", re.compile(r"(?:\+\d{1,3}[.\- ])?\b\d{2,4}[.\- ]\d{2,4}[.\- ]?\d{3,4}\b")),
    ("TAXNUM", re.compile(r"\b\d{5}[-\s]\d{5}\b")),
    ("PASSPORTNUM", re.compile(r"\b[A-Z]{2}\d{7}\b")),
    ("DRIVERLICENSENUM", re.compile(r"\b(?=[A-Z0-9]{10,13}\b)(?=.*[A-Z])(?=.*\d)[A-Z0-9]{10,13}\b")),
    ("IDCARDNUM", re.compile(r"\b(?=[A-Z0-9]{8,10}\b)(?=.*[A-Z])(?=.*\d)[A-Z0-9]{8,10}\b")),
    ("ZIPCODE", re.compile(r"\b\d{5}\b")),
]

WORD = re.compile(r"\b[\w-]+\b")

SPACY_LABELS = {"GPE": "CITY", "LOC": "CITY", "FAC": "STREET"}
"""spaCy type -> our label. CARDINAL is deliberately unmapped: it would have to guess
between AGE and BUILDINGNUM, and guessing costs precision on both. ORG has no counterpart."""


def predict(text: str) -> list[Span]:
    """Return non-overlapping spans, most specific pattern first."""
    claimed: list[tuple[int, int]] = []
    spans: list[Span] = []

    def free(start: int, end: int) -> bool:
        return not any(a < end and start < b for a, b in claimed)

    for label, pattern in PATTERNS:
        for m in pattern.finditer(text):
            if free(m.start(), m.end()):
                claimed.append((m.start(), m.end()))
                spans.append(Span(m.start(), m.end(), label))

    for m in WORD.finditer(text):
        token = m.group().lower()
        label = ("TITLE" if token in TITLES else
                 "GENDER" if token in GENDERS else
                 "SEX" if token in SEXES else None)
        if label and free(m.start(), m.end()):
            claimed.append((m.start(), m.end()))
            spans.append(Span(m.start(), m.end(), label))

    return sorted(spans, key=lambda s: s.start)


def predict_hybrid(text: str, nlp) -> list[Span]:
    """Patterns first, then spaCy NER for what patterns cannot reach."""
    spans = predict(text)
    claimed = [(s.start, s.end) for s in spans]

    def free(start: int, end: int) -> bool:
        return not any(a < end and start < b for a, b in claimed)

    for ent in nlp(text).ents:
        if ent.label_ == "PERSON":
            # Gold splits a name into TITLE / GIVENNAME / SURNAME; spaCy emits one span.
            # Approximate it by token position: first surviving token is the given name,
            # the remainder is the surname.
            tokens = [t for t in ent if t.text.lower().strip(".") not in TITLES and t.text.strip()]
            if not tokens:
                continue
            first = tokens[0]
            if free(first.idx, first.idx + len(first.text)):
                claimed.append((first.idx, first.idx + len(first.text)))
                spans.append(Span(first.idx, first.idx + len(first.text), "GIVENNAME"))
            if len(tokens) > 1:
                start, last = tokens[1].idx, tokens[-1]
                end = last.idx + len(last.text)
                if free(start, end):
                    claimed.append((start, end))
                    spans.append(Span(start, end, "SURNAME"))
        elif (label := SPACY_LABELS.get(ent.label_)) and free(ent.start_char, ent.end_char):
            claimed.append((ent.start_char, ent.end_char))
            spans.append(Span(ent.start_char, ent.end_char, label))

    return sorted(spans, key=lambda s: s.start)
