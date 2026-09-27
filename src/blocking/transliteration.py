"""Deterministic, offline Indic-to-Latin transliteration engine.

Provides rule-based, deterministic transliteration for the 9 major Brahmi-derived
Indic scripts present in the Amazon ML Challenge 2026 dataset:
- Devanagari (Hindi, Marathi)
- Bengali
- Gurmukhi (Punjabi)
- Gujarati
- Odia (Oriya)
- Tamil
- Telugu
- Kannada
- Malayalam

Uses exclusively the Python standard library and Unicode codepoint structure.
No external network, APIs, external databases, or pretrained models.
"""

from typing import Dict, Optional, Set

# Unicode block boundaries for supported Indic scripts
INDIC_START = 0x0900
INDIC_END = 0x0D7F

# Relative 128-codepoint offset mappings common across Brahmi-derived scripts
# Offset is computed as: (ord(char) & 0x7F)

OFFSET_VOWELS: Dict[int, str] = {
    0x01: "n",   # Chandrabindu
    0x02: "n",   # Anusvara (nasal)
    0x03: "h",   # Visarga
    0x05: "a",   # Independent A
    0x06: "aa",  # Independent AA
    0x07: "i",   # Independent I
    0x08: "ee",  # Independent II
    0x09: "u",   # Independent U
    0x0A: "oo",  # Independent UU
    0x0B: "ri",  # Independent Vocalic R
    0x0E: "e",   # Short E (Dravidian)
    0x0F: "e",   # Long E
    0x10: "ai",  # AI
    0x12: "o",   # Short O (Dravidian)
    0x13: "o",   # Long O
    0x14: "au",  # AU
}

OFFSET_CONSONANTS: Dict[int, str] = {
    0x15: "k",
    0x16: "kh",
    0x17: "g",
    0x18: "gh",
    0x19: "ng",
    0x1A: "ch",
    0x1B: "chh",
    0x1C: "j",
    0x1D: "jh",
    0x1E: "ny",
    0x1F: "t",
    0x20: "th",
    0x21: "d",
    0x22: "dh",
    0x23: "n",
    0x24: "t",
    0x25: "th",
    0x26: "d",
    0x27: "dh",
    0x28: "n",
    0x29: "n",   # Tamil NNNA (alveolar nasal)
    0x2A: "p",
    0x2B: "ph",
    0x2C: "b",
    0x2D: "bh",
    0x2E: "m",
    0x2F: "y",
    0x30: "r",
    0x31: "r",   # Tamil/Malayalam RRA
    0x32: "l",
    0x33: "l",   # Retroflex L
    0x34: "zh",  # Tamil/Malayalam LLLA (special retroflex approximant)
    0x35: "v",
    0x36: "sh",
    0x37: "sh",
    0x38: "s",
    0x39: "h",
    0x5C: "r",   # Bengali/Odia RRA (flapped)
    0x5D: "rh",  # Bengali/Odia RHA
    0x5F: "y",   # Bengali YYA
}

OFFSET_MATRAS: Dict[int, str] = {
    0x3E: "aa",
    0x3F: "i",
    0x40: "ee",
    0x41: "u",
    0x42: "oo",
    0x43: "ri",
    0x46: "e",
    0x47: "e",
    0x48: "ai",
    0x4A: "o",
    0x4B: "o",
    0x4C: "au",
}

VIRAMA_OFFSET = 0x4D


def has_indic_script(text: Optional[str]) -> bool:
    """Return True if text contains at least one Indic character (U+0900 to U+0D7F)."""
    if not text:
        return False
    for ch in text:
        if INDIC_START <= ord(ch) <= INDIC_END:
            return True
    return False


def transliterate_indic_to_latin(text: Optional[str]) -> str:
    """Deterministically convert Indic text to clean Latin ASCII representation.

    Rules:
    - Retains Latin ASCII, digits, punctuation, and whitespace verbatim.
    - Resolves Indic consonants with their attached vowel sign (matra) or virama (halant).
    - If a consonant is not followed by a matra or virama, an inherent 'a' is appended.
    - Vowels (independent and dependent) are normalized to phonetic English equivalents.
    - Script native digits (offsets 0x66-0x6F) are converted to standard 0-9 digits.
    """
    if not text:
        return ""

    out = []
    n = len(text)
    i = 0

    while i < n:
        ch = text[i]
        cp = ord(ch)

        # Check if in Indic script range (Devanagari through Malayalam)
        if INDIC_START <= cp <= INDIC_END:
            offset = cp & 0x7F

            # Case 1: Consonants
            if offset in OFFSET_CONSONANTS:
                cons = OFFSET_CONSONANTS[offset]
                # Look ahead for virama or matra
                if i + 1 < n and (INDIC_START <= ord(text[i + 1]) <= INDIC_END):
                    next_offset = ord(text[i + 1]) & 0x7F
                    if next_offset == VIRAMA_OFFSET:
                        # Virama silences inherent vowel
                        out.append(cons)
                        i += 2
                        continue
                    elif next_offset in OFFSET_MATRAS:
                        # Dependent vowel sign replaces inherent vowel
                        out.append(cons + OFFSET_MATRAS[next_offset])
                        i += 2
                        continue
                # Inherent 'a' vowel
                out.append(cons + "a")
                i += 1

            # Case 2: Independent vowels / nasals
            elif offset in OFFSET_VOWELS:
                out.append(OFFSET_VOWELS[offset])
                i += 1

            # Case 3: Native Indic Digits (offsets 0x66 to 0x6F map to 0..9)
            elif 0x66 <= offset <= 0x6F:
                out.append(str(offset - 0x66))
                i += 1

            # Case 4: Other script marks (e.g. nukta 0x3C, avagraha 0x3D) -> skip
            else:
                i += 1

        else:
            # Non-Indic character: retain as-is (Latin, ASCII digits, punctuation, space)
            out.append(ch)
            i += 1

    return "".join(out)
