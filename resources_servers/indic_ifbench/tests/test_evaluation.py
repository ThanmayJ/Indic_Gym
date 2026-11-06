# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkers, strict/loose scoring and the task contract."""

import json
from pathlib import Path

import pytest
from absl.testing import absltest

from resources_servers.indic_ifbench import languages
from resources_servers.indic_ifbench.evaluation import evaluate, get_instruction, instruction_kwargs
from resources_servers.indic_ifbench.languages import (
    MissingResource,
    count_words,
    split_sentences,
    word_pattern,
)
from resources_servers.indic_ifbench.task_data import TaskData, validate_instruction


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


class TokenizerTest(absltest.TestCase):
    def test_hindi_words_and_sentences(self):
        s = "हिन्दी भाषा में उत्तर दीजिए। यह एक परीक्षा है।"
        self.assertEqual(count_words(s), 9)
        self.assertLen(split_sentences(s, "hi"), 2)
        self.assertTrue(word_pattern("है").search("यह है।"))
        self.assertFalse(word_pattern("भाषा").search("भाषाएँ"))


class BucketBTest(absltest.TestCase):
    S = "हिन्दी भाषा में उत्तर दीजिए। यह एक परीक्षा है।"

    def test_word_count_and_keywords(self):
        self.assertTrue(check("count:word_count_range", self.S, {"min_words": 9, "max_words": 9}))
        self.assertTrue(check("sentence:keyword", self.S, {"word": "परीक्षा", "N": 2}))
        self.assertFalse(check("sentence:keyword", self.S, {"word": "परीक्षा", "N": 1}))
        self.assertFalse(
            check(
                "count:keywords_multiple",
                "तारा चाँद चाँद सूरज सूरज सूरज",
                {"keyword1": "तारा", "keyword2": "चाँद", "keyword3": "सूरज", "keyword4": "x", "keyword5": "y"},
            )
        )

    def test_sentences(self):
        self.assertTrue(check("ratio:sentence_type", "यह है। यह भी है। क्या यह है?", {}))
        self.assertTrue(check("words:last_first", "मैं घर जाता हूँ। हूँ मैं खुश। खुश रहो।", {}))
        self.assertTrue(check("words:paragraph_last_first", "घर अच्छा है घर।\nमैं जाता हूँ मैं।", {}))

    def test_options(self):
        self.assertTrue(check("format:options", "हाँ", {"options": "हाँ/नहीं/शायद"}))
        self.assertTrue(check("format:options", "मैं जानता हूँ", {"options": "मैं जानता हूँ या मैं नहीं जानता", **FIX}))


class BucketCTest(absltest.TestCase):
    def test_word_lists(self):
        self.assertTrue(check("count:pronouns", "मैं यह वह", {"N": 3}))
        self.assertTrue(check("count:conjunctions", "राम और श्याम लेकिन मोहन", {"small_n": 2}))
        self.assertTrue(check("ratio:stop_words", "राम घर जा रहा है", {"percentage": 25}))
        self.assertTrue(
            check("count:person_names", "राम और श्याम", {"N": 2, "_lang": {"names": ["राम", "श्याम", "मोहन"]}})
        )

    def test_fixed_strings(self):
        self.assertTrue(check("repeat:repeat_simple", "केवल यही वाक्य लिखें।", FIX))
        self.assertTrue(check("format:output_template", "मेरा उत्तर: क मेरा निष्कर्ष: ख भविष्य का दृष्टिकोण: ग", FIX))
        self.assertTrue(
            check(
                "custom:csv_city", "आईडी,देश,शहर,वर्ष,गिनती\n" + "\n".join("1,भारत,दिल्ली,2020,3" for _ in range(7)), FIX
            )
        )
        self.assertTrue(check("custom:word_reverse", "चील गंजा है", FIX))

    def test_missing_resource(self):
        with self.assertRaises(MissingResource):
            check("count:pronouns", "x", {"N": 1}, resources={})

    def test_script_rules(self):
        self.assertEqual(languages.brahmic_alphabet_ends("hi"), ("अ", "ह"))
        self.assertEqual(languages.brahmic_alphabet_ends("ta")[0], "அ")
        self.assertTrue(languages.brahmic_has_consonant_cluster("प्रश्न", "hi"))
        self.assertFalse(languages.brahmic_has_consonant_cluster("कमल", "hi"))
        self.assertEqual(languages.brahmic_vowel_types("कमल", "hi"), {5})
        self.assertEqual(languages.brahmic_vowel_types("किताब", "hi"), {5, 7, 6})
        self.assertEqual(languages.brahmic_count_syllables("किताब", "hi"), 3)
        self.assertTrue(check("words:consonants", "प्रश्न स्कूल", {}))
        self.assertFalse(check("words:consonants", "प्रश्न कमल", {}))
        self.assertTrue(check("words:vowel", "कमल कलम", {}))
        seq = languages.brahmic_alphabet_sequence("hi")
        self.assertTrue(check("words:alphabet", f"{seq[0]}क {seq[1]}ख {seq[2]}ग", {}))

    def test_sindhi_implosives_only_for_sindhi(self):
        self.assertTrue(languages.brahmic_has_consonant_cluster("ॻ्र", "sd"))
        self.assertFalse(languages.brahmic_has_consonant_cluster("ॻ्र", "hi"))
        with self.assertRaises(ValueError):
            languages.base_of("ur")  # Urdu has script rules, but not Brahmic ones


class RegistryTest(absltest.TestCase):
    def test_english_and_neutral_ids_use_upstream(self):
        self.assertTrue(check("count:numbers", "1 2 3", {"N": 3}))
        self.assertTrue(check("count:numbers", "1 2 3", {"N": 3}, lang="en"))
        self.assertEqual(type(get_instruction("count:word_count_range", "en")).__module__, "ifbench.instructions")
        self.assertTrue(
            type(get_instruction("count:word_count_range", "hi")).__module__.endswith("indic_ifbench.evaluation")
        )

    def test_float_indices_from_the_dataset(self):
        kw = {"prompt_to_repeat": "a b c d e", "n_start": 1.0, "n_end": 3.0}
        self.assertTrue(check("repeat:repeat_span", "b c d", kw))
        self.assertFalse(check("repeat:repeat_span", "b c", kw))


class DecisionsTest(absltest.TestCase):
    """One test per implemented decision in language-audit.md."""

    def test_c1_first_letter_is_first_character(self):
        self.assertFalse(check("words:no_consecutive", "किताब कागज़", {}))  # both start with the character क
        self.assertTrue(check("words:no_consecutive", "किताब मकान", {}))

    def test_c3_sentence_characters_are_unicode_characters(self):
        self.assertTrue(check("ratio:sentence_words", "कमल गया। रमन आया। नमन हुआ।", {}))  # 8 characters each
        self.assertFalse(check("ratio:sentence_words", "राम आया। श्याम गया। सीता गई।", {}))  # 8, 10, 9

    def test_d6_alphabet(self):
        seq = languages.brahmic_alphabet_sequence("hi")
        k = seq.index("क")
        words = [seq[k - 1] + "म", "क़लम", seq[k + 1] + "र"]  # क़ counts as क
        self.assertTrue(check("words:alphabet", " ".join(words), {}))
        sd = languages.brahmic_alphabet_sequence("sd")
        self.assertEqual(sd[sd.index("ग") + 1], "ॻ")
        self.assertEqual(sd[sd.index("ब") + 1], "ॿ")
        self.assertEqual(languages.brahmic_alphabet_ends("sd"), languages.brahmic_alphabet_ends("hi"))

    def test_d9_same_looking_text_matches(self):
        decomposed, composed = "क़लम", "क़लम"  # क + nukta vs precomposed क़
        self.assertTrue(check("sentence:keyword", f"यह {composed} है।", {"word": decomposed, "N": 1}))
        fixed = {"_lang": {"fixed": {"repeat_simple_sentence": "یہ جملہ لکھیں۔"}}}
        self.assertTrue(
            check("repeat:repeat_simple", "يہ جملہ لکھيں۔", fixed, lang="ur", resources={})
        )  # Arabic ي typed
        self.assertTrue(check("sentence:keyword", "क्‍या है।", {"word": "क्‍या", "N": 1}))  # joiners kept

    def test_d10a_native_full_stop_without_space(self):
        self.assertLen(split_sentences("पहला वाक्य।दूसरा वाक्य।", "hi"), 2)
        self.assertLen(split_sentences("کیا آپ؟ہاں۔", "ur"), 2)
        self.assertLen(split_sentences("तापमान 3.5 है।", "hi"), 1)  # "." still needs a space

    def test_d10b_pipe_as_danda(self):
        self.assertLen(split_sentences("पहला वाक्य | दूसरा वाक्य |", "hi"), 2)
        self.assertLen(split_sentences("| क | ख |\n| ग | घ |", "hi"), 1)  # Markdown table rows are not split
        self.assertLen(split_sentences("first | second |", "ta"), 1)  # not a Tamil boundary
        self.assertTrue(check("ratio:sentence_type", "यह है | यह भी है | क्या यह है?", {}))

    def test_d1_keyword_stem_and_suffix(self):
        for lang, kw, inflected in [
            ("ta", "புத்தகம்", "புத்தகத்தை"),
            ("ml", "പുസ്തകം", "പുസ്തകത്തെ"),
            ("te", "పుస్తకం", "పుస్తకాన్ని"),
            ("kn", "ಪುಸ್ತಕ", "ಪುಸ್ತಕವನ್ನು"),
        ]:
            self.assertTrue(
                check("sentence:keyword", f"நான் {inflected} படித்தேன்.", {"word": kw, "N": 1}, lang=lang, resources={}),
                lang,
            )
        self.assertFalse(
            check("sentence:keyword", "நான் மரத்தில் இருந்தேன்.", {"word": "மரம்", "N": 1}, lang="ta", resources={})
        )  # stem மர too short
        self.assertTrue(check("sentence:keyword", "நான் மரம் பார்த்தேன்.", {"word": "மரம்", "N": 1}, lang="ta", resources={}))
        self.assertFalse(
            check("sentence:keyword", "मैंने किताबें पढ़ीं।", {"word": "किताब", "N": 1})
        )  # Hindi: exact word only
        self.assertTrue(
            check(
                "words:words_position",
                "நான் புத்தகத்தை இன்று புத்தகம் படித்தேன்",
                {"keyword": "புத்தகம்"},
                lang="ta",
                resources={},
            )
        )

    def test_c9_reverse_by_unicode_characters(self):
        fixed = {"_lang": {"fixed": {"bald_eagle": "बाल्ड ईगल"}}}
        self.assertTrue(check("custom:character_reverse", "लगई ड्लाब", fixed))  # ब ा ल ् ड reversed: ड ् ल ा ब
        self.assertFalse(check("custom:character_reverse", "लगई ल्डबा", fixed))  # reversing letter units does not pass

    def test_d8_response_language(self):
        self.assertTrue(
            check(
                "response_language:en", "This is a short answer written in modern English for the test.", {}, lang="ta"
            )
        )
        self.assertFalse(check("response_language:en", "இது ஒரு தமிழ் பதில். இது சோதனைக்காக.", {}, lang="ta"))
        self.assertTrue(check("response_language:ru", "Это короткий ответ на русском языке.", {}, lang="hi"))
        from resources_servers.indic_ifbench import evaluation as evaluation_lib

        # a Tamil row that asks for an English answer: words:consonants is checked with the English (upstream) checker
        inp = evaluation_lib.InputExample(
            key=208, instruction_id_list=["words:consonants", "response_language:en"], prompt="p", kwargs=[{}, {}]
        )
        out = evaluation_lib.test_instruction_following_strict(
            inp, {"p": "Strong crisp bright plants grow."}, "ta", {}
        )
        self.assertEqual(out.follow_instruction_list, [True, True])

    def test_d10c_native_abbreviations(self):
        self.assertLen(split_sentences("திரு. ராமன் வந்தார். அவர் மகிழ்ச்சி.", "ta"), 2)
        self.assertLen(split_sentences("डॉ. शर्मा आए। वे खुश हैं।", "hi"), 2)

    def test_d10d_danda_in_marathi_and_konkani(self):
        self.assertLen(split_sentences("पहिले वाक्य। दुसरे वाक्य।", "mr"), 2)
        self.assertLen(split_sentences("पयलें वाक्य। दुसरें वाक्य।", "kok"), 2)


class ScriptRulesTest(absltest.TestCase):
    """D11: script rules for Urdu / Kashmiri (Perso-Arabic), Manipuri (Meetei Mayek), Santali (Ol Chiki)."""

    def test_perso_arabic(self):
        self.assertEqual(languages.alphabet_ends("ur"), ("ا", "ے"))
        self.assertEqual(languages.vowel_types("کتابیں", "ur"), {"aa", "ii"})  # written vowels only
        self.assertTrue(languages.has_consonant_cluster("کتاب", "ur"))  # ک and ت written next to each other
        self.assertFalse(languages.has_consonant_cluster("آؤ", "ur"))
        self.assertEqual(languages.count_syllables("کتابیں", "ur"), 2)
        self.assertEqual(languages.count_syllables("ہم", "ur"), 1)  # no written vowel: counts 1
        self.assertTrue(check("words:alphabet", "آپ بات پر", {}, lang="ur", resources={}))  # آ counts as ا
        self.assertEqual(languages.vowel_types("کٲشُر", "ks"), {"ə", "u"})  # Kashmiri vowel letter ٲ

    def test_meetei_mayek(self):
        self.assertEqual(languages.alphabet_ends("mni"), ("ꯀ", "ꯚ"))
        self.assertEqual(languages.vowel_types("ꯃꯤꯠꯌꯦꯡ", "mni"), {"i", "e"})
        self.assertEqual(languages.vowel_types("ꯀꯃꯜ", "mni"), {"a"})  # inherent a; final ꯜ has no vowel
        self.assertTrue(languages.has_consonant_cluster("ꯃꯤꯠꯌꯦꯡ", "mni"))  # final ꯠ + ꯌ
        self.assertTrue(languages.has_consonant_cluster("ꯀ꯭ꯔ", "mni"))  # consonant + apun iyek + consonant
        self.assertFalse(languages.has_consonant_cluster("ꯀꯃꯜ", "mni"))
        self.assertEqual(languages.count_syllables("ꯃꯤꯠꯌꯦꯡ", "mni"), 2)
        self.assertEqual(languages.vowel_types("ꯑꯩ", "mni"), {"ei"})  # ꯑ + vowel sign = that vowel

    def test_ol_chiki(self):
        self.assertEqual(languages.alphabet_ends("sat"), ("ᱚ", "ᱷ"))
        self.assertEqual(languages.vowel_types("ᱥᱟᱱᱛᱟᱲᱤ", "sat"), {"a", "i"})
        self.assertNotEqual(
            languages.vowel_types("ᱟᱹ", "sat"), languages.vowel_types("ᱟ", "sat")
        )  # ᱹ makes a new vowel
        self.assertTrue(languages.has_consonant_cluster("ᱥᱟᱱᱛᱟᱲᱤ", "sat"))  # ᱱᱛ
        self.assertFalse(languages.has_consonant_cluster("ᱟᱢ", "sat"))
        self.assertEqual(languages.count_syllables("ᱥᱟᱱᱛᱟᱲᱤ", "sat"), 3)


def test_strict_and_loose_decisions_per_instruction():
    strict, loose = evaluate(
        ["count:word_count_range", "words:prime_lengths"],
        [{"min_words": 2, "max_words": 3}, {}],
        language="hi",
        response="पानी\nराम है",
    )
    # पानी (4 characters) fails prime_lengths; the loose variant without the first line passes.
    assert strict == [True, False]
    assert loose == [True, True]


@pytest.mark.parametrize(("response", "expected"), [("राम है", True), ("राम पानी", False)])
def test_letters_are_single_unicode_characters(response, expected):
    # है = ह + ै is 2 characters (prime); पानी = प ा न ी is 4.
    assert evaluate(["words:prime_lengths"], [{}], language="hi", response=response)[0] == [expected]


def test_response_language_row_is_checked_in_the_answer_language():
    english = "The quick brown fox jumps over the lazy dog while the farmer watches from the distant hill."
    strict, _ = evaluate(["response_language:en"], [{}], language="ta", response=english)
    assert strict == [True]


@pytest.mark.parametrize(
    ("language", "response", "expected"), [("hi", "मैं और तुम", True), ("bn", "আমি আর তুমি", True), ("hi", "राम", False)]
)
def test_word_lists_are_loaded(language, response, expected):
    assert evaluate(["count:pronouns"], [{"N": 2}], language=language, response=response)[0] == [expected]


def test_empty_response_fails_every_instruction():
    assert evaluate(["count:pronouns"], [{"N": 1}], language="bn", response="") == ([False], [False])


def test_missing_language_resource_masks_instead_of_failing():
    kwargs = [{"_lang": {"fixed": {}}}]
    with pytest.raises(MissingResource):
        evaluate(["custom:character_reverse"], kwargs, language="hi", response="कुछ भी")


@pytest.mark.parametrize(("ids", "kwargs", "language"), [([], [], "hi"), (["count:pronouns"], [{"N": 1}], "xx")])
def test_invalid_input(ids, kwargs, language):
    with pytest.raises(ValueError):
        evaluate(ids, kwargs, language=language, response="text")


@pytest.mark.parametrize(
    ("instruction_id", "kwargs"), [("response_language:fr", {}), ("response_language:en", {"language": "en"})]
)
def test_response_language_validation(instruction_id, kwargs):
    with pytest.raises(ValueError):
        validate_instruction(instruction_id, kwargs, language="hi")


def test_response_language_task_validates():
    task = TaskData(
        id=1,
        language="bn",
        prompt="অনুবাদ করুন",
        instruction_id_list=["count:unique_word_count", "response_language:ru"],
        kwargs=[{"N": 3.0}, {}],
    )
    assert task.kwargs == [{"N": 3}, {}]


def test_committed_source_examples_score():
    path = Path(__file__).parents[1] / "data/example.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        task = TaskData.model_validate(json.loads(line))
        strict, loose = evaluate(task.instruction_id_list, task.kwargs, language=task.language, response="")
        assert strict == loose == [False] * len(task.instruction_id_list)
