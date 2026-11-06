# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Language data and text rules for Indic IFBench.

  Language data   data/languages/<lang>/config.json (sentence marks, punctuation, abbreviations, script) and
                  resources.json (pronouns, conjunctions, stop words, translated fixed strings)
  Text            Unicode-aware word and sentence splitting, punctuation, keywords, characters
  Script rules    alphabet, vowels, consonant clusters and syllables: Brahmic scripts by their Unicode block layout,
                  Perso-Arabic (Urdu, Kashmiri), Meetei Mayek (Manipuri) and Ol Chiki (Santali)

The conventions are listed in language-audit.md.
"""

import functools
import json
import os
import string
import unicodedata

import regex


# ---------------------------------------------------------------- language data

LANG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "languages")


def available():
    """Language codes that have a config.json (dataset languages and auxiliary answer languages such as ru)."""
    return sorted(d for d in os.listdir(LANG_DIR) if os.path.isfile(os.path.join(LANG_DIR, d, "config.json")))


def dataset_languages():
    """The 22 languages of the dataset (configs without `"auxiliary": true`)."""
    return [lang for lang in available() if not config(lang).get("auxiliary")]


@functools.lru_cache(maxsize=None)
def config(lang):
    path = os.path.join(LANG_DIR, lang, "config.json")
    if not os.path.isfile(path):
        raise KeyError(f"no language config for '{lang}' ({path})")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resources(lang):
    """Word lists and fixed strings for `lang` ({} if the language has no resources.json)."""
    path = os.path.join(LANG_DIR, lang, "resources.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_resources(lang, entry):
    os.makedirs(os.path.join(LANG_DIR, lang), exist_ok=True)
    with open(os.path.join(LANG_DIR, lang, "resources.json"), "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------- text

# Marks shared by all languages (ASCII and typographic); each language adds its own `punctuation` from its config.
COMMON_PUNCTUATION = string.punctuation + "‘’“”«»…–—·"


# ZWNJ / ZWJ (U+200C / U+200D) occur inside words in Indic and Perso-Arabic scripts and must not split them.
WORD_RE = regex.compile(r"[\p{L}\p{M}\p{N}_‌‍]+")


_NOT_WORD_BEFORE = r"(?<![\p{L}\p{M}\p{N}_])"


_NOT_WORD_AFTER = r"(?![\p{L}\p{M}\p{N}_])"


class MissingResource(Exception):
    """A language resource (word list or translated fixed string) needed by a checker is unavailable.
    The evaluator reports the instruction as skipped, not as failed."""


@functools.lru_cache(maxsize=None)
def punctuation(lang):
    """Characters treated as punctuation for `lang`: the common marks plus the language's own."""
    return COMMON_PUNCTUATION + "".join(config(lang)["punctuation"])


@functools.lru_cache(maxsize=None)
def _punctuation_table(lang):
    return {ord(c): None for c in punctuation(lang)}


# Sentence-final marks that only end a sentence when followed by whitespace ("." also occurs in 3.5 and abbreviations; "|" is
# accepted as a typed danda, decision D10b). Native full stops (danda, ۔, ؟, ꫰, ᱾ ...) end a sentence even without a following
# space, decision D10a.
_NEEDS_SPACE = set(".!?…|")


_CLOSERS = r"[\"'”’)\]]*"


@functools.lru_cache(maxsize=None)
def _sentence_boundary(lang):
    """One or more of the language's sentence-final marks, optional closing quotes/brackets, then whitespace (required only
    after ASCII marks)."""
    marks = config(lang)["sentence_end"]
    native = "".join(regex.escape(m) for m in marks if m not in _NEEDS_SPACE)
    ascii_ = "".join(regex.escape(m) for m in marks if m in _NEEDS_SPACE)
    parts = []
    if native:
        parts.append(r"[" + native + r"]+" + _CLOSERS + r"\s*")
    if ascii_:
        parts.append(r"[" + ascii_ + r"]+" + _CLOSERS + r"\s+")
    return regex.compile("|".join(parts))


def _in_table_row(text, pos):
    """True if `pos` lies on a line that starts with "|" (a Markdown table row): "|" there is not a danda."""
    line_start = text.rfind("\n", 0, pos) + 1
    line_end = text.find("\n", pos)
    return text[line_start : line_end if line_end >= 0 else len(text)].lstrip().startswith("|")


def normalize_text(text, lang):
    """Same-looking text, same code points (decision D9): Unicode NFC (nukta letters end up in one form) and the language's
    `char_variants` (Urdu / Kashmiri: Arabic-keyboard ي ى ك ه -> ی ی ک ہ). Joiners (ZWJ / ZWNJ) are kept."""
    text = unicodedata.normalize("NFC", text)
    variants = config(lang).get("char_variants")
    if variants:
        text = text.translate({ord(k): v for k, v in variants.items()})
    return text


@functools.lru_cache(maxsize=None)
def _abbreviations(lang):
    # A '.' after these tokens does not end a sentence (NLTK's English punkt also knows the English ones). Bare list
    # markers such as "1." DO split, matching upstream behaviour.
    return frozenset(config(lang)["abbreviations"])


def strip_punct(text, lang):
    return text.strip(punctuation(lang) + " ")


def remove_punct(text, lang):
    return text.translate(_punctuation_table(lang))


def split_words(text, lang):
    """Whitespace-delimited words with the language's edge punctuation stripped."""
    return [w for w in (strip_punct(t, lang) for t in text.split()) if w]


def count_words(text):
    """Counts word tokens, treating combining marks (vowel signs, viramas) and joiners as part of the word."""
    return len(WORD_RE.findall(text))


def word_tokens(text):
    return WORD_RE.findall(text)


def split_sentences(text, lang):
    """Sentences of `lang` text, split only at the language's own sentence-final marks (English: upstream splitter)."""
    if lang == "en":
        from ifbench import instructions_util

        return instructions_util.split_into_sentences(text)
    punct, abbreviations = punctuation(lang), _abbreviations(lang)
    text = text.strip()
    sentences, start = [], 0
    for m in _sentence_boundary(lang).finditer(text):
        if m.group(0)[0] == "|" and _in_table_row(text, m.start()):
            continue
        before = text[start : m.start()].split()
        last = before[-1].strip(punct).lower() if before else ""
        if m.group(0)[0] == "." and last in abbreviations:
            continue
        sentences.append(text[start : m.end()].strip())
        start = m.end()
    if text[start:].strip():
        sentences.append(text[start:].strip())
    return sentences


def sentence_type(sentence, lang):
    """'declarative', 'interrogative', 'exclamatory' or None, from the sentence-final mark (language config)."""
    cfg = config(lang)
    for kind in ("declarative", "interrogative", "exclamatory"):
        if sentence.endswith(tuple(cfg[f"{kind}_end"])):
            return kind
    return None


def word_pattern(word):
    """Regex matching `word` as a whole word (Unicode aware replacement for \\b...\\b)."""
    return regex.compile(_NOT_WORD_BEFORE + regex.escape(word) + _NOT_WORD_AFTER, regex.IGNORECASE)


MIN_KEYWORD_STEM = 4  # characters (code points); shorter stems would match unrelated words (decision D1)


def keyword_pattern(keyword, lang):
    """Regex for a keyword occurrence. In languages with `keyword_stem_matching` (ta, ml, te, kn; decision D1) the keyword's
    last letter unit may change and a suffix may follow (புத்தகம் matches புத்தகத்தை), provided the remaining stem has at least
    MIN_KEYWORD_STEM characters; otherwise, and in every other language, the keyword must occur as a whole word."""
    keyword = keyword.strip()
    if config(lang).get("keyword_stem_matching"):
        units = graphemes(keyword)
        stem = keyword[: len(keyword) - len(units[-1])] if len(units) > 1 else ""
        if len(stem) >= MIN_KEYWORD_STEM:
            return regex.compile(
                _NOT_WORD_BEFORE + regex.escape(stem) + r"[\p{L}\p{M}‌‍]*" + _NOT_WORD_AFTER, regex.IGNORECASE
            )
    return word_pattern(keyword)


def is_keyword(word, keyword, lang):
    """True if the single word `word` is an occurrence of `keyword` (same rule as keyword_pattern)."""
    return keyword_pattern(keyword, lang).fullmatch(word) is not None


def characters(text):
    """Characters as the benchmark counts them: single Unicode code points after NFC normalization, as "character"
    in English IFBench (decision 2026-11-07: a vowel sign or virama is a character of its own, पा = प + ा)."""
    return list(unicodedata.normalize("NFC", text))


def graphemes(word):
    """Letter units (extended grapheme clusters): a base letter with its vowel signs, virama joins and diacritics."""
    return regex.findall(r"\X", word)


# ---------------------------------------------------------------- Brahmic scripts

BLOCKS = {
    "devanagari": 0x0900,
    "bengali": 0x0980,
    "gurmukhi": 0x0A00,
    "gujarati": 0x0A80,
    "odia": 0x0B00,
    "tamil": 0x0B80,
    "telugu": 0x0C00,
    "kannada": 0x0C80,
    "malayalam": 0x0D00,
}


_VOWELS = range(0x05, 0x15)


_STANDARD_CONS = frozenset(range(0x15, 0x3A))


_NUKTA, _HALANT = 0x3C, 0x4D


_MATRA_TO_VOWEL = {
    0x3E: 0x06,
    0x3F: 0x07,
    0x40: 0x08,
    0x41: 0x09,
    0x42: 0x0A,
    0x43: 0x0B,
    0x44: 0x60,
    0x45: 0x0D,
    0x46: 0x0E,
    0x47: 0x0F,
    0x48: 0x10,
    0x49: 0x11,
    0x4A: 0x12,
    0x4B: 0x13,
    0x4C: 0x14,
}


_INHERENT_A = 0x05


_JOINERS = {"‌", "‍"}


def brahmic_has_script_rules(lang):
    return config(lang).get("script") in BLOCKS


def base_of(lang):
    script = config(lang).get("script")
    if script not in BLOCKS:
        raise ValueError(f"'{lang}' is not written in a Brahmic script (script: {script}); see script_rules.py")
    return BLOCKS[script]


@functools.lru_cache(maxsize=None)
def _consonants(lang):
    """Consonant offsets: the standard block range plus the language's `extra_consonants` (e.g. Sindhi implosives,
    which Unicode places at the end of the Devanagari block)."""
    base = base_of(lang)
    return _STANDARD_CONS | {ord(c) - base for c in config(lang).get("extra_consonants", [])}


def _offsets(word, base):
    """Offsets of the chars of `word` inside the script block (None for other chars)."""
    out = []
    for ch in word:
        if ch in _JOINERS:
            continue
        o = ord(ch) - base
        out.append((ch, o if 0 <= o < 0x80 else None))
    return out


def _is_letter(ch):
    return unicodedata.category(ch) == "Lo"


def brahmic_alphabet_sequence(lang):
    """Letters in Unicode block order (vowels, then consonants); `alphabet_after` in the language config places extra letters
    after a given letter (Sindhi: ॻ after ग, ॼ after ज, ॾ after ड, ॿ after ब), decision D6."""
    base = base_of(lang)
    seq = [chr(base + o) for o in range(0x05, 0x3A) if _is_letter(chr(base + o))]
    for letter, after in config(lang).get("alphabet_after", {}).items():
        seq.insert(seq.index(after) + 1, letter)
    return seq


def brahmic_alphabet_ends(lang):
    seq = brahmic_alphabet_sequence(lang)
    return seq[0], seq[-1]


def _next_off(offs, i):
    """Offset of the next char after position i, skipping nukta."""
    j = i + 1
    while j < len(offs) and offs[j][1] == _NUKTA:
        j += 1
    return offs[j][1] if j < len(offs) else None


def brahmic_vowel_types(word, lang):
    """Set of vowel identities used in the word (independent vowels, vowel signs, inherent 'a')."""
    base = base_of(lang)
    cons = _consonants(lang)
    offs = _offsets(word, base)
    vowels = set()
    for i, (ch, o) in enumerate(offs):
        if o is None:
            continue
        if o in _VOWELS and _is_letter(ch):
            vowels.add(o)
        elif o in _MATRA_TO_VOWEL:
            vowels.add(_MATRA_TO_VOWEL[o])
        elif o in cons and _is_letter(ch):
            n = _next_off(offs, i)
            if n not in _MATRA_TO_VOWEL and n != _HALANT:
                vowels.add(_INHERENT_A)
    return vowels


def brahmic_count_syllables(word, lang):
    """Orthographic syllables: one per vowel nucleus (no schwa deletion)."""
    base = base_of(lang)
    cons = _consonants(lang)
    offs = _offsets(word, base)
    n_syl = 0
    for i, (ch, o) in enumerate(offs):
        if o is None:
            continue
        if (o in _VOWELS and _is_letter(ch)) or o in _MATRA_TO_VOWEL:
            n_syl += 1
        elif o in cons and _is_letter(ch):
            nxt = _next_off(offs, i)
            if nxt not in _MATRA_TO_VOWEL and nxt != _HALANT:
                n_syl += 1
    return n_syl


def brahmic_has_consonant_cluster(word, lang):
    """True if the word contains consonant + virama + consonant (a conjunct)."""
    base = base_of(lang)
    cons = _consonants(lang)
    offs = [(c, o) for c, o in _offsets(word, base)]
    for i in range(len(offs) - 2):
        if offs[i][1] in cons:
            j = i + 1
            if offs[j][1] == _NUKTA:
                j += 1
            if j + 1 < len(offs) and offs[j][1] == _HALANT and offs[j + 1][1] in cons:
                return True
    return False


# ---------------------------------------------------------------- script rules


def script(lang):
    return config(lang).get("script")


def has_script_rules(lang):
    return bool(script(lang))


def _rules(lang):
    s = script(lang)
    if not s:
        raise ValueError(f"no script rules for '{lang}' (script is null in its config)")
    if s in BLOCKS:
        return _BRAHMIC
    return {"perso_arabic": _PERSO_ARABIC, "meetei_mayek": _MEETEI, "ol_chiki": _OL_CHIKI}[s]


def alphabet_sequence(lang):
    return _rules(lang).alphabet_sequence(lang)


def alphabet_ends(lang):
    seq = alphabet_sequence(lang)
    return seq[0], seq[-1]


def initial(word):
    """First letter of a word for the alphabet rule: canonical decomposition, so क़ counts as क and آ as ا."""
    return unicodedata.normalize("NFD", word)[0]


def vowel_types(word, lang):
    return _rules(lang).vowel_types(word, lang)


def has_consonant_cluster(word, lang):
    return _rules(lang).has_consonant_cluster(word, lang)


def count_syllables(word, lang):
    return _rules(lang).count_syllables(word, lang)


class _Brahmic:
    alphabet_sequence = staticmethod(brahmic_alphabet_sequence)
    vowel_types = staticmethod(brahmic_vowel_types)
    has_consonant_cluster = staticmethod(brahmic_has_consonant_cluster)
    count_syllables = staticmethod(brahmic_count_syllables)


class _PersoArabic:
    """Urdu, Kashmiri. Tables per language in config.json: `alphabet` (letter order) and `vowel_symbols`
    (letter or diacritic -> vowel it writes)."""

    @staticmethod
    def alphabet_sequence(lang):
        return list(config(lang)["alphabet"])

    @staticmethod
    def _symbols(lang):
        return config(lang)["vowel_symbols"]

    @staticmethod
    def _is_consonant(ch, lang):
        return ch in config(lang)["alphabet"] and ch not in config(lang)["vowel_symbols"]

    @classmethod
    def vowel_types(cls, word, lang):
        sym = cls._symbols(lang)
        return {sym[ch] for ch in unicodedata.normalize("NFD", word) if ch in sym}

    @classmethod
    def has_consonant_cluster(cls, word, lang):
        letters = [
            ch for ch in unicodedata.normalize("NFD", word) if unicodedata.category(ch) != "Mn"
        ]  # diacritics are transparent
        return any(cls._is_consonant(a, lang) and cls._is_consonant(b, lang) for a, b in zip(letters, letters[1:]))

    @classmethod
    def count_syllables(cls, word, lang):
        """Written vowels (vowel letters and vowel diacritics); a word with letters but no written vowel counts 1."""
        sym = cls._symbols(lang)
        n = sum(1 for ch in unicodedata.normalize("NFD", word) if ch in sym)
        return n if n else (1 if any(cls._is_consonant(ch, lang) for ch in word) else 0)


# ---------------------------------------------------------------- Meetei Mayek (U+ABC0-U+ABFF)
_MM_LETTERS = [chr(c) for c in range(0xABC0, 0xABDB)]  # ꯀ ... ꯚ, Unicode order = traditional order


_MM_INDEPENDENT_VOWELS = {"ꯑ": "a", "ꯏ": "i", "ꯎ": "u"}  # ꯑ (atiya, also the vowel carrier), ꯏ, ꯎ


_MM_LONSUM = {chr(c) for c in range(0xABDB, 0xABE3)}  # final consonants ꯛ ꯜ ꯝ ꯞ ꯟ ꯠ ꯡ ꯢ (no vowel)


_MM_VOWEL_SIGNS = {"ꯣ": "o", "ꯤ": "i", "ꯥ": "aa", "ꯦ": "e", "ꯧ": "ou", "ꯨ": "u", "ꯩ": "ei"}


_MM_APUN_IYEK = "꯭"  # virama


_MM_CONSONANTS = set(_MM_LETTERS) - set(_MM_INDEPENDENT_VOWELS)


class _Meetei:
    @staticmethod
    def alphabet_sequence(lang):
        return list(_MM_LETTERS)

    @staticmethod
    def _nuclei(word):
        """Vowel of each written syllable nucleus: vowel signs, independent vowels (ꯑ + sign = the sign's vowel),
        and the inherent 'a' of a consonant without vowel sign or apun iyek."""
        chars = list(word)
        out = []
        for i, ch in enumerate(chars):
            nxt = chars[i + 1] if i + 1 < len(chars) else ""
            if ch in _MM_VOWEL_SIGNS:
                out.append(_MM_VOWEL_SIGNS[ch])
            elif ch in _MM_INDEPENDENT_VOWELS:
                if nxt not in _MM_VOWEL_SIGNS:
                    out.append(_MM_INDEPENDENT_VOWELS[ch])
            elif ch in _MM_CONSONANTS and nxt not in _MM_VOWEL_SIGNS and nxt != _MM_APUN_IYEK:
                out.append("a")
        return out

    @classmethod
    def vowel_types(cls, word, lang):
        return set(cls._nuclei(word))

    @staticmethod
    def has_consonant_cluster(word, lang):
        chars = list(word)
        for i in range(len(chars) - 1):
            if (
                chars[i] in _MM_CONSONANTS
                and chars[i + 1] == _MM_APUN_IYEK
                and i + 2 < len(chars)
                and chars[i + 2] in _MM_CONSONANTS
            ):
                return True
            if chars[i] in _MM_LONSUM and (chars[i + 1] in _MM_CONSONANTS or chars[i + 1] in _MM_LONSUM):
                return True
        return False

    @classmethod
    def count_syllables(cls, word, lang):
        return len(cls._nuclei(word))


# ---------------------------------------------------------------- Ol Chiki (U+1C50-U+1C7F)
_OC_LETTERS = [chr(c) for c in range(0x1C5A, 0x1C78)]  # ᱚ ... ᱷ, Unicode order = traditional order


_OC_VOWELS = {"ᱚ": "o", "ᱟ": "a", "ᱤ": "i", "ᱩ": "u", "ᱮ": "e", "ᱳ": "ó"}  # ᱚ ᱟ ᱤ ᱩ ᱮ ᱳ


_OC_GAAHLAA = "ᱹ"  # ᱹ: changes the vowel (ᱟᱹ = ə)


_OC_MODIFIERS = {chr(c) for c in range(0x1C78, 0x1C7E)}  # ᱸ ᱹ ᱺ ᱻ ᱼ ᱽ (transparent between letters)


_OC_CONSONANTS = set(_OC_LETTERS) - set(_OC_VOWELS)


class _OlChiki:
    @staticmethod
    def alphabet_sequence(lang):
        return list(_OC_LETTERS)

    @staticmethod
    def vowel_types(word, lang):
        """Vowel letters; a vowel followed by gaahlaa ttuddaag (ᱹ) is a different vowel type."""
        out = set()
        for i, ch in enumerate(word):
            if ch in _OC_VOWELS:
                out.add(_OC_VOWELS[ch] + ("̈" if word[i + 1 : i + 2] == _OC_GAAHLAA else ""))
        return out

    @staticmethod
    def has_consonant_cluster(word, lang):
        letters = [ch for ch in word if ch not in _OC_MODIFIERS]
        return any(a in _OC_CONSONANTS and b in _OC_CONSONANTS for a, b in zip(letters, letters[1:]))

    @staticmethod
    def count_syllables(word, lang):
        return sum(1 for ch in word if ch in _OC_VOWELS)


_BRAHMIC, _PERSO_ARABIC, _MEETEI, _OL_CHIKI = _Brahmic, _PersoArabic, _Meetei, _OlChiki
