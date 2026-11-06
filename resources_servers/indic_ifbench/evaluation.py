# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scoring for Indic IFBench: the IFBench checkers adapted per language.

Each checker subclasses the upstream AllenAI IFBench checker (cloned at a pinned commit by setup_ifbench.py) and
replaces the English-specific parts with the language's rules from languages.py: word and sentence splitting, word
lists, script rules, and the translated fixed strings and person names in the task's kwargs `_lang`. Instructions
without English-specific logic use the upstream checker unchanged. The conventions are listed in language-audit.md.
"""

import csv
import dataclasses
import functools
import io
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Dict, Optional, Union

import langdetect
import regex
from pydantic import JsonValue

from resources_servers.indic_ifbench.setup_ifbench import ensure_ifbench


ensure_ifbench()  # the checkers subclass the upstream `ifbench` package

from ifbench import classic_instructions as _ifb_classic  # noqa: E402
from ifbench import instructions as _ifb  # noqa: E402
from ifbench import instructions_registry as _ifb_registry  # noqa: E402

from resources_servers.indic_ifbench import languages  # noqa: E402
from resources_servers.indic_ifbench.languages import (  # noqa: E402
    MissingResource,
    characters,
    count_words,
    is_keyword,
    keyword_pattern,
    normalize_text,
    punctuation,
    remove_punct,
    sentence_type,
    split_sentences,
    split_words,
    strip_punct,
    word_tokens,
)
from resources_servers.indic_ifbench.task_data import EVALUATOR_VERSION  # noqa: E402, F401


# ---------------------------------------------------------------- checkers

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}


def _i(x):
    return int(float(x))


def _normalized(v, lang):
    """kwargs strings (keywords, options, operands, fixed strings, names) get the same normalization as the response (D9)."""
    if isinstance(v, str):
        return normalize_text(v, lang)
    if isinstance(v, list):
        return [_normalized(x, lang) for x in v]
    if isinstance(v, dict):
        return {k: _normalized(x, lang) for k, x in v.items()}
    return v


class MultilingualInstruction:
    """Mixin placed before the upstream class: language-aware construction and kwargs handling."""

    def __init__(self, instruction_id, lang, resources=None):
        super().__init__(instruction_id)
        self.lang = lang
        self.resources = resources or {}
        self._kwargs = {}

    def __init_subclass__(cls, **kwargs):
        """Every check_following defined by a subclass first normalizes the response (decision D9)."""
        super().__init_subclass__(**kwargs)
        check = cls.__dict__.get("check_following")
        if check is not None:

            def check_following(self, value, _check=check):
                return _check(self, normalize_text(value, self.lang))

            check_following.__doc__ = check.__doc__
            cls.check_following = check_following

    def build_description(self, **kwargs):
        self._kwargs.update({k: _normalized(v, self.lang) for k, v in kwargs.items() if v is not None})
        return None

    def get_instruction_args(self):
        return {k: v for k, v in self._kwargs.items() if k != "_lang"}

    def get_instruction_args_keys(self):
        return list(self.get_instruction_args())

    def _fixed(self, key):
        v = ((self._kwargs.get("_lang") or {}).get("fixed") or {}).get(key)
        if not v:
            raise MissingResource(f"fixed string '{key}'")
        return v

    def _list_resource(self, key):
        v = self.resources.get(key)
        if not v:
            raise MissingResource(f"{self.lang}.{key}")
        return _normalized(v, self.lang)  # word lists get the same normalization as the response (D9)


class WordCountRangeChecker(MultilingualInstruction, _ifb.WordCountRangeChecker):
    """count:word_count_range: word tokens keep vowel signs, virama and joiners inside the word."""

    def check_following(self, value):
        return _i(self._kwargs["min_words"]) <= count_words(value) <= _i(self._kwargs["max_words"])


class UniqueWordCountChecker(MultilingualInstruction, _ifb.UniqueWordCountChecker):
    """count:unique_word_count"""

    def check_following(self, value):
        return len({w.lower() for w in split_words(value, self.lang)}) >= _i(self._kwargs["N"])


class KeywordsMultipleChecker(MultilingualInstruction, _ifb.KeywordsMultipleChecker):
    """count:keywords_multiple: keyword1..5 must occur exactly 1, 2, 3, 5, 7 times as whole words."""

    def check_following(self, value):
        for key, count in zip(("keyword1", "keyword2", "keyword3", "keyword4", "keyword5"), (1, 2, 3, 5, 7)):
            if len(keyword_pattern(self._kwargs[key], self.lang).findall(value)) != count:
                return False
        return True


class KeywordSpecificPositionChecker(MultilingualInstruction, _ifb.KeywordSpecificPositionChecker):
    """words:keywords_specific_position: keyword is the m-th word of the n-th sentence."""

    def check_following(self, value):
        n, m = _i(self._kwargs["n"]), _i(self._kwargs["m"])
        sents = split_sentences(value, self.lang)
        if len(sents) < n:
            return False
        words = split_words(sents[n - 1], self.lang)
        return len(words) >= m and is_keyword(words[m - 1], self._kwargs["keyword"], self.lang)


class WordsPositionChecker(MultilingualInstruction, _ifb.WordsPositionChecker):
    """words:words_position: keyword is the second word and the second-to-last word."""

    def check_following(self, value):
        words = split_words(value, self.lang)
        return (
            len(words) >= 2
            and is_keyword(words[1], self._kwargs["keyword"], self.lang)
            and is_keyword(words[-2], self._kwargs["keyword"], self.lang)
        )


class IncludeKeywordChecker(MultilingualInstruction, _ifb.IncludeKeywordChecker):
    """sentence:keyword: the N-th sentence contains the keyword as a whole word."""

    def check_following(self, value):
        n = _i(self._kwargs["N"])
        sents = split_sentences(value, self.lang)
        return len(sents) >= n and bool(keyword_pattern(self._kwargs["word"], self.lang).search(sents[n - 1]))


class IncrementingWordCountChecker(MultilingualInstruction, _ifb.IncrementingWordCountChecker):
    """sentence:increment: each sentence has exactly small_n more words than the previous one."""

    def check_following(self, value):
        sents = split_sentences(value, self.lang)
        if not sents:
            return False
        prev = len(split_words(sents[0], self.lang))
        for s in sents[1:]:
            cur = len(split_words(s, self.lang))
            if cur != prev + _i(self._kwargs["small_n"]):
                return False
            prev = cur
        return True


class IncrementingAlliterationChecker(MultilingualInstruction, _ifb.IncrementingAlliterationChecker):
    """sentence:alliteration_increment: each sentence has more alliterating words than the previous (first code point)."""

    def check_following(self, value):
        prev_all = -1
        for s in split_sentences(value, self.lang):
            words = [w.lstrip(punctuation(self.lang) + " ") for w in s.lower().split()]
            words = [w for w in words if w]
            alliteration, prev_alliterative = 0, False
            for i in range(len(words) - 1):
                if words[i][0] == words[i + 1][0]:
                    alliteration += 1 if prev_alliterative else 2
                    prev_alliterative = True
                else:
                    prev_alliterative = False
            if alliteration <= prev_all:
                return False
            prev_all = alliteration
        return True


def _sentence_type_counts(value, lang):
    types = [sentence_type(s, lang) for s in split_sentences(value, lang)]
    return types.count("declarative"), types.count("interrogative"), types.count("exclamatory")


class SentTypeRatioChecker(MultilingualInstruction, _ifb.SentTypeRatioChecker):
    """ratio:sentence_type: declarative : interrogative = 2 : 1."""

    def check_following(self, value):
        d, q, _ = _sentence_type_counts(value, self.lang)
        return d == 2 * q


class SentBalanceChecker(MultilingualInstruction, _ifb.SentBalanceChecker):
    """ratio:sentence_balance: equal numbers of declarative, interrogative and exclamatory sentences."""

    def check_following(self, value):
        d, q, e = _sentence_type_counts(value, self.lang)
        return d == q == e


class CharacterCountUniqueWordsChecker(MultilingualInstruction, _ifb.CharacterCountUniqueWordsChecker):
    """ratio:sentence_words: 3 sentences of equal length in letter units (decision D5), no repeated word."""

    def check_following(self, value):
        sents = split_sentences(value, self.lang)
        if len(sents) != 3:
            return False
        n = len(characters(sents[0].strip()))  # characters are single Unicode characters (C3)
        if any(len(characters(s.strip())) != n for s in sents):
            return False
        words = [w.lower() for w in split_words(" ".join(sents), self.lang)]
        return len(words) == len(set(words))


class EmojiSentenceChecker(MultilingualInstruction, _ifb.EmojiSentenceChecker):
    """format:emoji: every sentence ends with an emoji (before the final punctuation)."""

    def check_following(self, value):
        import emoji

        sents = split_sentences(value, self.lang)
        for i, s in enumerate(sents):
            stripped = remove_punct(s, self.lang).strip()
            if not stripped:
                return False
            last = stripped[-1]
            second = stripped[-2] if len(stripped) > 1 else stripped[-1]
            if not emoji.is_emoji(last) and not emoji.is_emoji(second):
                if i < len(sents) - 1:
                    nxt = remove_punct(sents[i + 1], self.lang).strip()
                    if not nxt or not emoji.is_emoji(nxt[0]):
                        return False
                else:
                    return False
        return True


class SomeBulletPointsChecker(MultilingualInstruction, _ifb.SomeBulletPointsChecker):
    """format:no_bullets_bullets: at least two sentences, then at least two '*' bullets."""

    def check_following(self, value):
        sentences, n_sent, n_bul = True, 0, 0
        for line in value.split("\n"):
            if line.strip().startswith("*"):
                sentences = False
                if n_sent < 2:
                    return False
                n_bul += 1
            elif sentences:
                sentences = split_sentences(line.strip(), self.lang)
                n_sent += len(sentences)
            else:
                return False
        return n_bul >= 2


class LastWordFirstNextChecker(MultilingualInstruction, _ifb.LastWordFirstNextChecker):
    """words:last_first: the last word of each sentence is the first word of the next."""

    def check_following(self, value):
        sents = split_sentences(value, self.lang)
        for a, b in zip(sents, sents[1:]):
            punct = punctuation(self.lang) + " "
            last, first = a.rstrip(punct).split(), b.lstrip(punct).split()
            if (
                not last
                or not first
                or strip_punct(last[-1], self.lang).lower() != strip_punct(first[0], self.lang).lower()
            ):
                return False
        return True


class ParagraphLastFirstWordMatchChecker(MultilingualInstruction, _ifb.ParagraphLastFirstWordMatchChecker):
    """words:paragraph_last_first: each paragraph starts and ends with the same word."""

    def check_following(self, value):
        for p in value.split("\n"):
            words = strip_punct(p.strip().lower(), self.lang).split()
            if words and strip_punct(words[0], self.lang) != strip_punct(words[-1], self.lang):
                return False
        return True


class NoConsecutiveFirstLetterChecker(MultilingualInstruction, _ifb.NoConsecutiveFirstLetterChecker):
    """words:no_consecutive: no two adjacent words start with the same character (C1: first Unicode character, so कि
    and का share their first letter क)."""

    def check_following(self, value):
        words = [w.lower() for w in split_words(value, self.lang)]
        firsts = [characters(w)[0] for w in words]
        return all(firsts[i] != firsts[i + 1] for i in range(len(firsts) - 1))


class LimitedWordRepeatChecker(MultilingualInstruction, _ifb.LimitedWordRepeatChecker):
    """words:repeats: no word occurs more than small_n times."""

    def check_following(self, value):
        from collections import Counter

        return all(
            c <= _i(self._kwargs["small_n"])
            for c in Counter(w.lower() for w in split_words(value, self.lang)).values()
        )


class NGramOverlapChecker(MultilingualInstruction, _ifb.NGramOverlapChecker):
    """ratio:overlap: word-trigram overlap with the reference text within percentage +/- 2."""

    def check_following(self, value):
        def grams(t):
            w = split_words(t, self.lang)
            return set(zip(w, w[1:], w[2:]))

        g = grams(value)
        if not g:
            return False
        pct = len(g & grams(self._kwargs["reference_text"])) / len(g) * 100
        return _i(self._kwargs["percentage"]) - 2 <= pct <= _i(self._kwargs["percentage"]) + 2


class PalindromeChecker(MultilingualInstruction, _ifb.PalindromeChecker):
    """words:palindrome: at least 10 palindromes of at least 5 characters (single Unicode characters, C2)."""

    def check_following(self, value):
        n = 0
        for w in split_words(value, self.lang):
            g = characters(w.lower())
            if len(g) >= 5 and g == g[::-1]:
                n += 1
        return n >= 10


class PrimeLengthsChecker(MultilingualInstruction, _ifb.PrimeLengthsChecker):
    """words:prime_lengths: every word has a prime number of characters (single Unicode characters, C2)."""

    def check_following(self, value):
        return all(len(characters(w)) in PRIMES for w in split_words(value, self.lang))


class OptionsResponseChecker(MultilingualInstruction, _ifb.OptionsResponseChecker):
    """format:options: the response is exactly one of the (translated) options; "or" is the translated word."""

    def check_following(self, value):
        opts = self._kwargs["options"]
        strict = regex.match(r"\W*[aA]\W*[bB]\W*[cC]\W*", opts) is not None
        if "/" in opts:
            sep = "/"
        else:
            or_word = ((self._kwargs.get("_lang") or {}).get("fixed", {}).get("or_word")) or "or"
            sep = f" {or_word} " if f" {or_word} " in opts else ","
        parts = [o.strip() for o in opts.split(sep)]
        if strict:
            return value in parts
        v = strip_punct(value, self.lang).lower()
        return any(strip_punct(o, self.lang).lower() == v for o in parts)


class MultipleChoiceQuestionsChecker(MultilingualInstruction, _ifb.MultipleChoiceQuestionsChecker):
    """custom:mcq_count_length: 4 questions with the translated label, 5 options each, increasing question length."""

    def check_following(self, value):
        label = self._fixed("question_label")
        if not value.startswith(label):
            return False
        questions = regex.split(r"\n*(?:" + regex.escape(label) + r"\s*\p{N}+[\.|\):;]?\s*)", value)
        if questions and questions[0] == "":
            questions = questions[1:]
        questions = [q.strip() for q in questions if q.strip()]
        if len(questions) != 4:
            return False
        lengths = []
        for q in questions:
            text, opts, done = "", 0, False
            for line in q.split("\n"):
                if regex.match(r"^\p{L}\p{M}*[\.|\)]\s*\S+", line.strip()):
                    opts += 1
                    done = True
                elif not done:
                    text += " " + line.strip()
            if opts != 5:
                return False
            lengths.append(len(text.strip()))
        return all(a < b for a, b in zip(lengths, lengths[1:]))


class PronounCountChecker(MultilingualInstruction, _ifb.PronounCountChecker):
    """count:pronouns: at least N pronoun tokens from the language's pronoun list."""

    def check_following(self, value):
        pron = {p.lower() for p in self._list_resource("pronouns")}
        toks = [t.lower() for t in word_tokens(value.replace("/", " "))]
        return sum(1 for t in toks if t in pron) >= _i(self._kwargs["N"])


class ConjunctionCountChecker(MultilingualInstruction, _ifb.ConjunctionCountChecker):
    """count:conjunctions: at least small_n distinct conjunctions from the language's list."""

    def check_following(self, value):
        conj = {c.lower() for c in self._list_resource("conjunctions")}
        found = {w.lower() for w in split_words(value, self.lang) if w.lower() in conj}
        return len(found) >= _i(self._kwargs["small_n"])


class StopWordPercentageChecker(MultilingualInstruction, _ifb.StopWordPercentageChecker):
    """ratio:stop_words: stop words (language list) are at most `percentage` % of the tokens."""

    def check_following(self, value):
        sw = {s.lower() for s in self._list_resource("stopwords")}
        toks = [t.lower() for t in word_tokens(value)]
        if not toks:
            return False
        return sum(1 for t in toks if t in sw) / len(toks) * 100 <= float(self._kwargs["percentage"])


class PersonNameCountChecker(MultilingualInstruction, _ifb.PersonNameCountChecker):
    """count:person_names: at least N distinct names from the transliterated list in `_lang.names`."""

    def check_following(self, value):
        names = (self._kwargs.get("_lang") or {}).get("names")
        if not names:
            raise MissingResource("person names list")
        found = {
            n
            for n in names
            if regex.search(r"(?<![\p{L}\p{M}\p{N}_])" + regex.escape(n) + r"(?![\p{L}\p{M}\p{N}_])", value)
        }
        return len(found) >= _i(self._kwargs["N"])


class AlphabetLoopChecker(MultilingualInstruction, _ifb.AlphabetLoopChecker):
    """words:alphabet: word initials follow the script's letter sequence (Unicode block order, decision D6), looping."""

    def check_following(self, value):
        seq = languages.alphabet_sequence(self.lang)
        words = split_words(value, self.lang)
        initial = languages.initial  # D6: क़ counts as क (and آ as ا)
        if not words or initial(words[0]) not in seq:
            return False
        idx = seq.index(initial(words[0]))
        for w in words[1:]:
            idx = (idx + 1) % len(seq)
            if initial(w) != seq[idx]:
                return False
        return True


class SingleVowelParagraphChecker(MultilingualInstruction, _ifb.SingleVowelParagraphChecker):
    """words:vowel: one paragraph using at most three vowel types (vowel letters, vowel signs, inherent 'a')."""

    def check_following(self, value):
        paras = value.strip().split("\n")
        if len(paras) != 1:
            return False
        vowels = set()
        for w in paras[0].split():
            vowels |= languages.vowel_types(w, self.lang)
        return len(vowels) <= 3


class ConsonantClusterChecker(MultilingualInstruction, _ifb.ConsonantClusterChecker):
    """words:consonants: every word contains a conjunct (consonant + virama + consonant)."""

    def check_following(self, value):
        words = split_words(value, self.lang)
        return bool(words) and all(languages.has_consonant_cluster(w, self.lang) for w in words)


class AlternateParitySyllablesChecker(MultilingualInstruction, _ifb.AlternateParitySyllablesChecker):
    """words:odd_even_syllables: adjacent words alternate odd/even orthographic syllable counts."""

    def check_following(self, value):
        syl = [languages.count_syllables(w, self.lang) % 2 for w in split_words(value, self.lang)]
        return all(syl[i] != syl[i + 1] for i in range(len(syl) - 1))


class RepeatSimpleChecker(MultilingualInstruction, _ifb.RepeatSimpleChecker):
    """repeat:repeat_simple: the response is exactly the translated sentence."""

    def check_following(self, value):
        return value.strip().lower() == self._fixed("repeat_simple_sentence").strip().lower()


class WordReverseOrderChecker(MultilingualInstruction, _ifb.WordReverseOrderChecker):
    """custom:word_reverse: reading the words backwards gives the translated answer."""

    def check_following(self, value):
        v = remove_punct(value.lower().strip(), self.lang)
        v = " ".join(v.split()[::-1])
        return self._fixed("bald_eagle").lower() in v


class CharacterReverseOrderChecker(MultilingualInstruction, _ifb.CharacterReverseOrderChecker):
    """custom:character_reverse: the translated answer reversed character by character (C9: single Unicode
    characters)."""

    def check_following(self, value):
        return "".join(characters(self._fixed("bald_eagle").lower())[::-1]) in value.lower()


class EuropeanCapitalsSortChecker(MultilingualInstruction, _ifb.EuropeanCapitalsSortChecker):
    """custom:european_capitals_sort: the translated capital names, in the expected order, comma separated."""

    def check_following(self, value):
        order = [unicodedata.normalize("NFC", c).casefold() for c in self._fixed("european_capitals")]
        caps = [unicodedata.normalize("NFC", c).strip().casefold() for c in value.split(",") if c.strip()]
        return caps == order


class OutputTemplateChecker(MultilingualInstruction, _ifb.OutputTemplateChecker):
    """format:output_template: the three translated labels are present."""

    def check_following(self, value):
        return all(self._fixed(k) in value for k in ("my_answer", "my_conclusion", "future_outlook"))


def _csv_header_ok(value, cols, sep=","):
    pat = r"^" + (sep + r"[ \t]*").join(f'(?:{regex.escape(c)}|"{regex.escape(c)}")' for c in cols) + r"$"
    return regex.match(pat, value.split("\n")[0].strip()) is not None


class CityCSVChecker(MultilingualInstruction, _ifb.CityCSVChecker):
    """custom:csv_city: translated header + 7 data rows."""

    def check_following(self, value):
        cols = self._fixed("csv_city_cols")
        data = list(csv.reader(io.StringIO(value)))
        return len(data) == 8 and data[0] == cols and all(len(r) == len(cols) for r in data[1:])


class SpecialCharacterCSVChecker(MultilingualInstruction, _ifb.SpecialCharacterCSVChecker):
    """custom:csv_special_character: translated header, 14 data rows, a quoted field with a special character."""

    def check_following(self, value):
        cols = self._fixed("csv_special_cols")
        if not _csv_header_ok(value, cols):
            return False
        data = list(csv.reader(io.StringIO(value.replace('"', '"""'))))
        if len(data) != 15:
            return False
        for row in data[1:]:
            if len(row) != len(cols):
                return False
            if any(regex.match(r'".*[^\p{L}\p{M}\p{N}_\s].*"', f) for f in row):
                return True
        return False


class QuotesCSVChecker(MultilingualInstruction, _ifb.QuotesCSVChecker):
    """custom:csv_quotes: tab separated, translated header, 3 data rows, every field quoted."""

    def check_following(self, value):
        cols = self._fixed("csv_quotes_cols")
        header = value.split("\n")[0].strip()
        pat = r"^" + r"\t *".join(f'(?:{regex.escape(c)}|"{regex.escape(c)}")' for c in cols) + r"$"
        if regex.match(pat, header) is None:
            return False
        data = list(csv.reader(io.StringIO(value.replace('"', '"""')), delimiter="\t"))
        if len(data) != 4:
            return False
        return all(
            len(r) == len(cols) and all(f.strip()[:1] == '"' and f.strip()[-1:] == '"' for f in r) for r in data
        )


langdetect.DetectorFactory.seed = 0  # langdetect is random on short texts unless seeded: keep results reproducible


class ResponseLanguageChecker(_ifb_classic.ResponseLanguageChecker):
    """response_language:xx: the answer must be in language xx (detected with langdetect, as IFBench's
    language:response_language). Added to rows that ask for an answer in another language than the dataset language
    (e.g. a translation into English); its presence also makes the evaluator check that row's other instructions with
    language xx (decision D8)."""

    PREFIX = "response_language:"

    def __init__(self, instruction_id, lang=None, resources=None):
        super().__init__(instruction_id)

    def build_description(self, **kwargs):
        return super().build_description(language=self.id[len(self.PREFIX) :])

    @classmethod
    def language_of(cls, instruction_id_list):
        """The answer language required by a row, or None."""
        for i in instruction_id_list:
            if i.startswith(cls.PREFIX):
                return i[len(cls.PREFIX) :]
        return None


# ---------------------------------------------------------------- registry

# IDs whose checker depends on the language; every other ID uses the upstream (language neutral) class.
MULTILINGUAL_INSTRUCTION_DICT = {
    "count:word_count_range": WordCountRangeChecker,
    "count:unique_word_count": UniqueWordCountChecker,
    "count:keywords_multiple": KeywordsMultipleChecker,
    "words:keywords_specific_position": KeywordSpecificPositionChecker,
    "words:words_position": WordsPositionChecker,
    "sentence:keyword": IncludeKeywordChecker,
    "sentence:increment": IncrementingWordCountChecker,
    "sentence:alliteration_increment": IncrementingAlliterationChecker,
    "ratio:sentence_type": SentTypeRatioChecker,
    "ratio:sentence_balance": SentBalanceChecker,
    "ratio:sentence_words": CharacterCountUniqueWordsChecker,
    "format:emoji": EmojiSentenceChecker,
    "format:no_bullets_bullets": SomeBulletPointsChecker,
    "words:last_first": LastWordFirstNextChecker,
    "words:paragraph_last_first": ParagraphLastFirstWordMatchChecker,
    "words:no_consecutive": NoConsecutiveFirstLetterChecker,
    "words:repeats": LimitedWordRepeatChecker,
    "ratio:overlap": NGramOverlapChecker,
    "words:palindrome": PalindromeChecker,
    "words:prime_lengths": PrimeLengthsChecker,
    "format:options": OptionsResponseChecker,
    "custom:mcq_count_length": MultipleChoiceQuestionsChecker,
    "count:pronouns": PronounCountChecker,
    "count:conjunctions": ConjunctionCountChecker,
    "ratio:stop_words": StopWordPercentageChecker,
    "count:person_names": PersonNameCountChecker,
    "words:alphabet": AlphabetLoopChecker,
    "words:vowel": SingleVowelParagraphChecker,
    "words:consonants": ConsonantClusterChecker,
    "words:odd_even_syllables": AlternateParitySyllablesChecker,
    "repeat:repeat_simple": RepeatSimpleChecker,
    "custom:word_reverse": WordReverseOrderChecker,
    "custom:character_reverse": CharacterReverseOrderChecker,
    "custom:european_capitals_sort": EuropeanCapitalsSortChecker,
    "format:output_template": OutputTemplateChecker,
    "custom:csv_city": CityCSVChecker,
    "custom:csv_special_character": SpecialCharacterCSVChecker,
    "custom:csv_quotes": QuotesCSVChecker,
}


# Answer-language markers (decision D8): one ID per language that the dataset asks answers in.
RESPONSE_LANGUAGE_IDS = {f"response_language:{code}": ResponseLanguageChecker for code in ("en", "ru")}


INSTRUCTION_DICT = {**_ifb_registry.INSTRUCTION_DICT, **MULTILINGUAL_INSTRUCTION_DICT, **RESPONSE_LANGUAGE_IDS}


def get_instruction(instruction_id, lang, resources=None):
    """Checker instance for a response in `lang`. English responses, and IDs without a multilingual class, use the
    upstream IFBench class unchanged."""
    if instruction_id in RESPONSE_LANGUAGE_IDS:
        return RESPONSE_LANGUAGE_IDS[instruction_id](instruction_id)
    if lang != "en" and instruction_id in MULTILINGUAL_INSTRUCTION_DICT:
        return MULTILINGUAL_INSTRUCTION_DICT[instruction_id](instruction_id, lang=lang, resources=resources)
    return _ifb_registry.INSTRUCTION_DICT[instruction_id](instruction_id)


def instruction_kwargs(instruction, kwargs):
    """Row kwargs for `build_description`: null values dropped; integer-valued floats (the dataset stores e.g. 10.0)
    passed as int, since upstream checkers use them as list indices; `_lang` kept only for the multilingual classes
    (the upstream classes do not accept it)."""
    kw = {
        k: int(v) if isinstance(v, float) and v.is_integer() else v for k, v in (kwargs or {}).items() if v is not None
    }
    if not isinstance(instruction, MultilingualInstruction):
        kw.pop("_lang", None)
    return kw


# ---------------------------------------------------------------- strict and loose evaluation


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: Optional[bool]  # None when an instruction was skipped
    follow_instruction_list: list[Optional[bool]]  # True / False / None (skipped: missing language resource)


def empty_output(inp):
    """Builds a failed output for prompts with no generated response."""
    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response="",
        follow_all_instructions=False,
        follow_instruction_list=[False] * len(inp.instruction_id_list),
    )


def _row_language(inp, lang, resources):
    """Language (and resources) for checking this row: the dataset language, unless the row requires its answer in
    another language through a response_language:xx instruction (decision D8)."""
    answer_lang = ResponseLanguageChecker.language_of(inp.instruction_id_list)
    if not answer_lang or answer_lang == lang:
        return lang, resources
    return answer_lang, (languages.resources(answer_lang) if answer_lang in languages.available() else {})


def _build_instruction(inp, index, lang, resources):
    lang, resources = _row_language(inp, lang, resources)
    instruction_id = inp.instruction_id_list[index]
    instruction = get_instruction(instruction_id, lang, resources)
    instruction.build_description(**instruction_kwargs(instruction, inp.kwargs[index]))
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
        instruction.build_description(prompt=inp.prompt)
    return instruction


def _output(inp, response, is_following_list):
    skipped = any(x is None for x in is_following_list)
    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=None if skipped else all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_strict(inp, prompt_to_response, lang, resources=None):
    """Tests response to see if instructions are followed."""
    response = get_response_for_prompt(inp, prompt_to_response)
    if response is None:
        return empty_output(inp)

    is_following_list = []
    for index in range(len(inp.instruction_id_list)):
        instruction = _build_instruction(inp, index, lang, resources)
        try:
            is_following_list.append(bool(response and response.strip() and instruction.check_following(response)))
        except MissingResource:
            is_following_list.append(None)
    return _output(inp, response, is_following_list)


def test_instruction_following_loose(inp, prompt_to_response, lang, resources=None):
    """Tests response for an upper bound for following instructions."""
    response = get_response_for_prompt(inp, prompt_to_response)
    if response is None:
        return empty_output(inp)

    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    all_responses = [
        response,
        response.replace("*", ""),
        response_remove_first,
        response_remove_last,
        response_remove_both,
        response_remove_first.replace("*", ""),
        response_remove_last.replace("*", ""),
        response_remove_both.replace("*", ""),
    ]

    is_following_list = []
    for index in range(len(inp.instruction_id_list)):
        instruction = _build_instruction(inp, index, lang, resources)
        try:
            is_following = any(c.strip() and instruction.check_following(c) for c in all_responses)
            is_following_list.append(bool(is_following))
        except MissingResource:
            is_following_list.append(None)
    return _output(inp, response, is_following_list)


def get_response_for_prompt(inp, prompt_to_response):
    """Returns a prompt response, tolerating harmless leading/trailing spaces."""
    if inp.prompt in prompt_to_response:
        return prompt_to_response[inp.prompt]
    return prompt_to_response.get(inp.prompt.strip())


# ---------------------------------------------------------------- Gym entry point


@functools.lru_cache(maxsize=None)
def _resources(language: str) -> dict:
    """The language's word lists (pronouns, conjunctions, stop words) from data/languages/<lang>/resources.json."""
    return languages.resources(language)


def evaluate(
    instruction_ids: Sequence[str],
    kwargs: Sequence[Mapping[str, JsonValue]],
    *,
    language: str,
    response: str,
    prompt: str = "",
) -> tuple[list[bool], list[bool]]:
    """Per-instruction strict and loose decisions for one response.

    An instruction whose language resource (word list or translated fixed string) is missing cannot be scored; this
    raises MissingResource so that the sample is masked instead of counted as a model failure.
    """
    if not instruction_ids or len(instruction_ids) != len(kwargs):
        raise ValueError("instruction_id_list and kwargs must be nonempty and have equal lengths")
    if language not in languages.available():
        raise ValueError(f"Unsupported language {language!r}")
    example = InputExample(
        key=0, instruction_id_list=list(instruction_ids), prompt=prompt, kwargs=[dict(k) for k in kwargs]
    )
    responses = {prompt: response}
    resources = _resources(language)
    strict = test_instruction_following_strict(example, responses, language, resources)
    loose = test_instruction_following_loose(example, responses, language, resources)
    strict, loose = strict.follow_instruction_list, loose.follow_instruction_list
    missing = [i for i, value in zip(instruction_ids, strict) if value is None]
    if missing:
        raise MissingResource(f"no {language} resources for {missing}")
    return [bool(v) for v in strict], [bool(v) for v in loose]
