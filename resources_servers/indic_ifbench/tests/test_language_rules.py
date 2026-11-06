# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-language configs (sentence marks, punctuation) and per-language checker cases.

Cases: (description, instruction_id, response, kwargs, expected); expected None = informational only.
"""

from absl.testing import parameterized

from resources_servers.indic_ifbench import languages
from resources_servers.indic_ifbench.evaluation import get_instruction, instruction_kwargs
from resources_servers.indic_ifbench.languages import (
    split_sentences,
    split_words,
)


HI_RESOURCES = {
    "pronouns": ["मैं", "यह", "वह"],
    "conjunctions": ["और", "लेकिन", "या"],
    "stopwords": ["है", "यह", "में", "और"],
}
FIX = {
    "_lang": {
        "fixed": {
            "question_label": "प्रश्न",
            "my_answer": "मेरा उत्तर:",
            "my_conclusion": "मेरा निष्कर्ष:",
            "future_outlook": "भविष्य का दृष्टिकोण:",
            "repeat_simple_sentence": "केवल यही वाक्य लिखें।",
            "bald_eagle": "गंजा चील",
            "european_capitals": ["ओस्लो", "पेरिस"],
            "or_word": "या",
            "csv_city_cols": ["आईडी", "देश", "शहर", "वर्ष", "गिनती"],
        }
    }
}


def check(instruction_id, value, kwargs, lang="hi", resources=None):
    """Instantiate, build and check exactly as evaluation.py does."""
    instruction = get_instruction(
        instruction_id, lang, HI_RESOURCES if resources is None and lang == "hi" else resources
    )
    instruction.build_description(**instruction_kwargs(instruction, dict(kwargs)))
    return instruction.check_following(value)


# ---- hi
HI_FIX = {"_lang": {"fixed": {"question_label": "प्रश्न", "or_word": "या"}}}
HI_ZWNJ, HI_ZWJ = "\u200c", "\u200d"

HI_CASES = [
    # --- word_count_range (upstream: \w+ tokens) ---
    ("wc: plain 4 words", "count:word_count_range", "मेरा नाम राम है।", {"min_words": 4, "max_words": 4}, True),
    (
        "wc: matras/conjuncts must not split words",
        "count:word_count_range",
        "हिन्दी भाषा सुन्दर है",
        {"min_words": 4, "max_words": 4},
        True,
    ),
    (
        "wc: hyphenated counts as 2 (as upstream)",
        "count:word_count_range",
        "राम-श्याम आए",
        {"min_words": 3, "max_words": 3},
        True,
    ),
    (
        "wc: word containing HI_ZWNJ stays 1 word",
        "count:word_count_range",
        f"क्{HI_ZWNJ}या है",
        {"min_words": 2, "max_words": 2},
        True,
    ),
    (
        "wc: word containing HI_ZWJ stays 1 word",
        "count:word_count_range",
        f"क्{HI_ZWJ}या है",
        {"min_words": 2, "max_words": 2},
        True,
    ),
    ("wc: Devanagari digits", "count:word_count_range", "मेरे पास १२३ किताबें हैं", {"min_words": 5, "max_words": 5}, True),
    ("wc: too long fails", "count:word_count_range", "एक दो तीन चार पाँच", {"min_words": 1, "max_words": 4}, False),
    # --- sentence splitting ---
    (
        "sent: danda split + keyword in 2nd",
        "sentence:keyword",
        "यह पहला वाक्य है। यहाँ चाँद है।",
        {"word": "चाँद", "N": 2},
        True,
    ),
    ("sent: ? and ! split", "sentence:keyword", "क्या तुम आए? हाँ! चाँद निकला।", {"word": "चाँद", "N": 3}, True),
    (
        "sent: abbreviation 'डॉ.' splits wrongly (known)",
        "sentence:keyword",
        "डॉ. शर्मा आए। चाँद निकला।",
        {"word": "चाँद", "N": 2},
        True,
    ),
    (
        "sent: decimal 3.5 must not split",
        "sentence:keyword",
        "तापमान 3.5 डिग्री है। चाँद निकला।",
        {"word": "चाँद", "N": 2},
        True,
    ),
    (
        "sent: ellipsis ends a sentence (as nltk usually does)",
        "sentence:keyword",
        "मैं सोचता हूँ... चाँद निकला।",
        {"word": "चाँद", "N": 2},
        True,
    ),
    ("sent: quote closes after danda", "sentence:keyword", "उसने कहा “जाओ।” चाँद निकला।", {"word": "चाँद", "N": 2}, True),
    (
        "sent: list numbering '1. ' splits like upstream",
        "sentence:keyword",
        "सूची: 1. चाँद 2. सूरज",
        {"word": "सूरज", "N": 3},
        True,
    ),
    # --- keywords with word boundaries ---
    ("kw: keyword ending in matra + danda", "words:words_position", "वह है राम।", {"keyword": "है"}, True),
    (
        "kw: keyword with HI_ZWNJ",
        "words:words_position",
        f"वह क्{HI_ZWNJ}या राम क्{HI_ZWNJ}या है",
        {"keyword": f"क्{HI_ZWNJ}या"},
        True,
    ),
    (
        "kw: oblique form must not count",
        "count:keywords_multiple",
        "चाँद सूरज सूरज तारा तारा तारा आम आम आम आम आम केला केला केला केला केला केला केला",
        {"keyword1": "चाँद", "keyword2": "सूरज", "keyword3": "तारा", "keyword4": "आम", "keyword5": "केला"},
        True,
    ),
    (
        "kw: substring inside longer word not counted",
        "count:keywords_multiple",
        "चाँदनी सूरज सूरज तारा तारा तारा आम आम आम आम आम केला केला केला केला केला केला केला",
        {"keyword1": "चाँद", "keyword2": "सूरज", "keyword3": "तारा", "keyword4": "आम", "keyword5": "केला"},
        False,
    ),
    ("pos: 2nd word and 2nd-to-last", "words:words_position", "मैं सुंदर हूँ सुंदर जी", {"keyword": "सुंदर"}, True),
    (
        "pos: keyword as m-th word of n-th sentence",
        "words:keywords_specific_position",
        "एक वाक्य यहाँ है। मेरा घर बड़ा है।",
        {"keyword": "घर", "n": 2, "m": 2},
        True,
    ),
    # --- sentence types / balance ---
    ("type: 2 declarative : 1 interrogative", "ratio:sentence_type", "यह घर है। वह पेड़ है। क्या तुम आए?", {}, True),
    ("type: exclamation breaks ratio test", "ratio:sentence_type", "यह घर है। क्या तुम आए? वाह!", {}, False),
    ("balance: one of each", "ratio:sentence_balance", "यह घर है। क्या तुम आए? वाह!", {}, True),
    ("balance: ascii period also counts", "ratio:sentence_balance", "This is a house. क्या तुम आए? वाह!", {}, True),
    # --- emoji / bullets ---
    ("emoji: emoji before danda", "format:emoji", "मैं खुश हूँ 😀। वह भी खुश है 😀।", {}, True),
    ("emoji: missing on second", "format:emoji", "मैं खुश हूँ 😀। वह भी खुश है।", {}, False),
    ("emoji: variation selector heart", "format:emoji", "मुझे प्यार है ❤️। वह भी ❤️।", {}, True),
    (
        "bullets: 2 sentences then 2 bullets",
        "format:no_bullets_bullets",
        "यह पहला वाक्य है। यह दूसरा वाक्य है।\n* पहला\n* दूसरा",
        {},
        True,
    ),
    ("bullets: only one bullet", "format:no_bullets_bullets", "यह पहला वाक्य है। यह दूसरा वाक्य है।\n* पहला", {}, False),
    # --- last/first word ---
    ("last_first: valid 3-sentence chain", "words:last_first", "राम घर गया। गया वह खेल। खेल अच्छा है।", {}, True),
    ("last_first: broken chain fails", "words:last_first", "राम घर गया। वह खेल।", {}, False),
    ("last_first: valid", "words:last_first", "वह घर गया। गया सुबह। सुबह अच्छी थी।", {}, True),
    ("para_last_first: danda stripped", "words:paragraph_last_first", "घर अच्छा है घर।\nमैं जाता हूँ मैं।", {}, True),
    ("para_last_first: mismatch", "words:paragraph_last_first", "घर अच्छा है।\nमैं जाता हूँ मैं।", {}, False),
    # --- palindrome / prime lengths (single Unicode characters) ---
    ("pal: 10 palindromes len>=5", "words:palindrome", " ".join(["कनकनक"] * 10), {}, True),
    ("pal: 9 only fails", "words:palindrome", " ".join(["कनकनक"] * 9), {}, False),
    ("prime: lengths 2,3,5", "words:prime_lengths", "राम कमल रमानाथ", {}, None),
    ("prime: है = ह ै is 2 characters, prime", "words:prime_lengths", "राम है", {}, True),
    ("prime: पानी = प ा न ी is 4 characters, fails", "words:prime_lengths", "राम पानी", {}, False),
    # --- options ---
    ("opt: yes/no/maybe", "format:options", "हाँ", {"options": "हाँ/नहीं/शायद"}, True),
    ("opt: with danda/punct", "format:options", "नहीं।", {"options": "हाँ/नहीं/शायद"}, True),
    ("opt: explanation fails", "format:options", "हाँ, मुझे लगता है", {"options": "हाँ/नहीं/शायद"}, False),
    ("opt: translated 'or'", "format:options", "मुझे नहीं पता", {"options": "मुझे पता है या मुझे नहीं पता", **HI_FIX}, True),
    ("opt: strict letters", "format:options", "b)", {"options": "a), b), c), d)"}, True),
    # --- mcq ---
    (
        "mcq: 4 questions, 5 options each, growing",
        "custom:mcq_count_length",
        "प्रश्न 1. क्या है?\nA. एक\nB. दो\nC. तीन\nD. चार\nE. पाँच\n"
        "प्रश्न 2. यह क्या चीज़ है भाई?\nA. एक\nB. दो\nC. तीन\nD. चार\nE. पाँच\n"
        "प्रश्न 3. बीसवीं सदी की कला का इतिहास क्या है भाई?\nA. एक\nB. दो\nC. तीन\nD. चार\nE. पाँच\n"
        "प्रश्न 4. बीसवीं सदी की कला का इतिहास किस आंदोलन से शुरू होता है भाई?\nA. एक\nB. दो\nC. तीन\nD. चार\nE. पाँच",
        HI_FIX,
        True,
    ),
    (
        "mcq: Devanagari option letters",
        "custom:mcq_count_length",
        "प्रश्न 1. क्या है?\nक. एक\nख. दो\nग. तीन\nघ. चार\nङ. पाँच\n"
        "प्रश्न 2. यह क्या चीज़ है भाई?\nक. एक\nख. दो\nग. तीन\nघ. चार\nङ. पाँच\n"
        "प्रश्न 3. बीसवीं सदी की कला का इतिहास क्या है भाई?\nक. एक\nख. दो\nग. तीन\nघ. चार\nङ. पाँच\n"
        "प्रश्न 4. बीसवीं सदी की कला का इतिहास किस आंदोलन से शुरू होता है भाई?\nक. एक\nख. दो\nग. तीन\nघ. चार\nङ. पाँच",
        HI_FIX,
        True,
    ),
    (
        "mcq: preamble before first question fails",
        "custom:mcq_count_length",
        "ये रहे प्रश्न:\nप्रश्न 1. क्या है?\nA. a\nB. b\nC. c\nD. d\nE. e",
        HI_FIX,
        False,
    ),
    # --- alliteration (first char) ---
    ("allit: increasing", "sentence:alliteration_increment", "राम आया। कमल कपड़ा कल लाया।", {}, True),
    # --- sentence_words ---
    ("sentence_words: 3 equal length, distinct", "ratio:sentence_words", "राम आया। श्याम गया। मोहन सोया।", {}, None),
]

# ---- mni
MNI_CASES = [
    ("sent: cheikhei U+AAF0 ends a sentence", "sentence:keyword", "ꯑꯩ ꯂꯥꯛꯂꯦ꫰ ꯃꯍꯥꯛ ꯆꯠꯂꯦ꫰", {"word": "ꯃꯍꯥꯛ", "N": 2}, True),
    ("sent: U+ABEB also ends a sentence", "sentence:keyword", "ꯑꯩ ꯂꯥꯛꯂꯦ꯫ ꯃꯍꯥꯛ ꯆꯠꯂꯦ꯫", {"word": "ꯃꯍꯥꯛ", "N": 2}, True),
    (
        "sent: a danda is not a Manipuri boundary",
        "sentence:keyword",
        "ꯑꯩ ꯂꯥꯛꯂꯦ। ꯃꯍꯥꯛ ꯆꯠꯂꯦ꫰",
        {"word": "ꯃꯍꯥꯛ", "N": 2},
        False,
    ),
    ("type: 2 statements : 1 question", "ratio:sentence_type", "ꯑꯩ ꯂꯥꯛꯂꯦ꫰ ꯃꯍꯥꯛ ꯆꯠꯂꯦ꫰ ꯅꯍꯥꯛ ꯂꯥꯛꯂꯤꯕꯔꯥ? ", {}, True),
    ("last_first: full stop stripped from the last word", "words:last_first", "ꯑꯩ ꯂꯥꯛꯂꯦ꫰ ꯂꯥꯛꯂꯦ ꯃꯍꯥꯛ꫰", {}, True),
]

# ---- sat
SAT_CASES = [
    ("sent: mucaad U+1C7E ends a sentence", "sentence:keyword", "ᱥᱟᱹᱜᱩᱱ ᱾ ᱟᱢ ᱾ ᱪᱮᱫ? ", {"word": "ᱟᱢ", "N": 2}, True),
    ("sent: a danda is not a Santali boundary", "sentence:keyword", "ᱥᱟᱹᱜᱩᱱ। ᱟᱢ ᱾", {"word": "ᱟᱢ", "N": 2}, False),
    ("type: 2 statements : 1 question", "ratio:sentence_type", "ᱥᱟᱹᱜᱩᱱ ᱾ ᱟᱢ ᱾ ᱪᱮᱫ? ", {}, True),
    ("balance: one of each", "ratio:sentence_balance", "ᱥᱟᱹᱜᱩᱱ ᱾ ᱪᱮᱫ? ᱟᱢ!", {}, True),
]

LANGUAGE_CASES = {"hi": HI_CASES, "mni": MNI_CASES, "sat": SAT_CASES}
LANGUAGE_RESOURCES = {}


def _language_cases():
    out = []
    for lang, cases in LANGUAGE_CASES.items():
        resources = LANGUAGE_RESOURCES.get(lang, {})
        for i, (desc, inst, resp, kw, expected) in enumerate(cases):
            if expected is not None:
                out.append((f"{lang}_{i:02d}", lang, inst, resp, kw, expected, resources))
    return out


ALL_MARKS = sorted({m for lang in languages.available() for m in languages.config(lang)["sentence_end"]})


class LanguageConfigTest(parameterized.TestCase):
    def test_every_language_has_a_folder(self):
        try:
            from ifbench_config import BRAHMIC_LANGS, LANG_MAP  # the translation pipeline's config (not in NeMo Gym)
        except ImportError:
            self.skipTest("ifbench_config (translation pipeline) not available")
        self.assertEqual(set(languages.dataset_languages()), set(LANG_MAP))
        self.assertEqual(
            {lang for lang in languages.dataset_languages() if languages.config(lang)["script"] in languages.BLOCKS},
            BRAHMIC_LANGS,
        )
        self.assertTrue(
            all(languages.has_script_rules(lang) for lang in languages.dataset_languages())
        )  # D11: all 22 have rules
        for lang in languages.dataset_languages():
            self.assertTrue(languages.resources(lang), f"{lang}: resources.json missing or empty")

    @parameterized.named_parameters((lang, lang) for lang in languages.available())
    def test_only_own_sentence_marks(self, lang):
        own = set(languages.config(lang)["sentence_end"])
        for mark in ALL_MARKS:
            n = len(split_sentences(f"aa bb{mark} cc dd", lang))
            self.assertEqual(n, 2 if mark in own else 1, f"{lang}: mark {mark!r}")

    @parameterized.named_parameters((lang, lang) for lang in languages.available())
    def test_only_own_punctuation_stripped(self, lang):
        own = set(languages.config(lang)["punctuation"])
        for mark in sorted({m for lang in languages.available() for m in languages.config(lang)["punctuation"]}):
            self.assertEqual(
                split_words(f"ab{mark}", lang), ["ab"] if mark in own else [f"ab{mark}"], f"{lang}: {mark!r}"
            )

    def test_language_is_required(self):
        with self.assertRaises(TypeError):
            get_instruction("count:word_count_range", "hi").__class__("count:word_count_range")


class LanguageCasesTest(parameterized.TestCase):
    """Cases from languages/<lang>/test_cases.py."""

    @parameterized.named_parameters(_language_cases())
    def test_case(self, lang, inst, resp, kw, expected, res):
        self.assertEqual(check(inst, resp, kw, lang=lang, resources=res), expected)
