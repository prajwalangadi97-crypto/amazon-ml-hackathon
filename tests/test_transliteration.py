"""Unit tests for deterministic Indic-to-Latin transliteration."""

import pytest
from src.blocking.transliteration import has_indic_script, transliterate_indic_to_latin


def test_has_indic_script():
    # Empty / None
    assert not has_indic_script("")
    assert not has_indic_script(None)

    # Latin only
    assert not has_indic_script("Amazon ML Challenge 2026")
    assert not has_indic_script("123 Main Street, Suite 400")

    # Indic scripts
    assert has_indic_script("தமிழ்")  # Tamil
    assert has_indic_script("हिन्दी")  # Devanagari
    assert has_indic_script("বাংলা")  # Bengali
    assert has_indic_script("ગુજરાતી")  # Gujarati
    assert has_indic_script("ଓଡ଼ିଆ")  # Odia
    assert has_indic_script("తెలుగు")  # Telugu
    assert has_indic_script("ಕನ್ನಡ")  # Kannada
    assert has_indic_script("മലയാളം")  # Malayalam
    assert has_indic_script("ਪੰਜਾਬੀ")  # Gurmukhi

    # Mixed
    assert has_indic_script("Raj Investments ராஜ்")
    assert has_indic_script("Plot 123/4 ওଡ଼ିଆ Cuttack")


def test_transliterate_latin_invariance():
    # Non-Indic text should remain completely unchanged
    assert transliterate_indic_to_latin("Amazon ML Challenge") == "Amazon ML Challenge"
    assert transliterate_indic_to_latin("123/4, 5th Avenue, NY 10001") == "123/4, 5th Avenue, NY 10001"
    assert transliterate_indic_to_latin("") == ""
    assert transliterate_indic_to_latin(None) == ""


def test_transliterate_devanagari():
    # 'राज' -> r + aa + j + a (or raj)
    t = transliterate_indic_to_latin("राज")
    assert t.startswith("raaj")

    # Conjunct with virama: 'स्टेट' (State) -> s + virama + t + e + t -> stet
    t = transliterate_indic_to_latin("स्टेट")
    assert "stet" in t


def test_transliterate_tamil():
    # 'ராஜ்' (Raj with virama on Ja) -> r + aa + j = raaj
    t = transliterate_indic_to_latin("ராஜ்")
    assert t == "raaj"

    # 'சென்னை' (Chennai) -> ch + e + n + n + ai
    t = transliterate_indic_to_latin("சென்னை")
    assert "chenn" in t or "sen" in t


def test_transliterate_odia():
    # 'ରେଡ୍' (Red with virama on Da) -> r + e + d = red
    t = transliterate_indic_to_latin("ରେଡ୍")
    assert t == "red"

    # 'ଏନର୍ଜି' (Energy)
    t = transliterate_indic_to_latin("ଏନର୍ଜି")
    assert "enar" in t


def test_transliterate_bengali():
    # 'কলকাতা' (Kolkata) -> k + l + k + aa + t + aa
    t = transliterate_indic_to_latin("কলকাতা")
    assert "kalakaataa" in t or "kalkaata" in t


def test_transliterate_gujarati():
    # 'સુરત' (Surat) -> s + u + r + a + t
    t = transliterate_indic_to_latin("સુરત")
    assert "surat" in t


def test_transliterate_telugu():
    # 'హైదరాబాద్' (Hyderabad)
    t = transliterate_indic_to_latin("హైదరాబాద్")
    assert "haidar" in t


def test_transliterate_kannada():
    # 'ಬೆಂಗಳೂರು' (Bengaluru)
    t = transliterate_indic_to_latin("ಬೆಂಗಳೂರು")
    assert "bengal" in t or "benga" in t


def test_transliterate_real_failure_cases():
    # Real validation failure 1: 'ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி' (Raj Investments LLP)
    t1 = transliterate_indic_to_latin("ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி")
    assert "raaj" in t1
    assert "elelpi" in t1 or "llp" in t1 or "el" in t1

    # Real validation failure 2: 'ରେଡ୍ ଏନର୍ଜି ପ୍ରାଇଭେଟ୍ ଲିମିଟେଡ୍' (Red Energy Pvt Ltd)
    t2 = transliterate_indic_to_latin("ରେଡ୍ ଏନର୍ଜି ପ୍ରାଇଭେଟ୍ ଲିମିଟେଡ୍")
    assert "red" in t2
    assert "enarji" in t2
    assert "limited" in t2 or "limite" in t2
