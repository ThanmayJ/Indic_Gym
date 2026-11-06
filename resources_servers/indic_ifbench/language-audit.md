# Language audit: evaluator decisions

The checkers decide whether a model's answer followed an instruction. For Indian languages some instructions can be checked in
more than one reasonable way, and the choice changes scores. Each item below is **one question for you**.

## How to answer

1. Read **What happens now** (a real example from the current code).
2. Pick one option and write it after **Your answer:**, e.g. `Your answer: B`. Add a note if you want a variation.
   The option marked *(recommended)* is my suggestion; you can choose any.
3. Nothing changes until you answer. Then each answer is implemented with its own tests.

Part 1 can be decided now. Part 2 is better decided after seeing real model answers (Phase 4), because it depends on how
models actually write. Part 3 is done and needs nothing from you.

## Conventions adopted (from the answers in Part 1)

These are the definitions the Indian-language checkers use. They differ from a naive reading of the English instructions,
so cite them when reporting results.

| # | Convention | Decision | Instruction |
|---|---|---|---|
| C1 | The first letter of a word is its first **Unicode character**, so कि and का share their first letter क (decision 2026-11-07, replaces letter units; the prompts carry no definition). | D2 | words:no_consecutive |
| C2 | Word length is counted in **single Unicode characters** (code points after NFC): है = ह + ै has length 2, पानी has length 4 (decision 2026-11-07, replaces letter units). | D3 | words:prime_lengths, words:palindrome |
| C3 | "Characters" in a sentence are **single Unicode characters** (spaces and punctuation count as one each) (decision 2026-11-07). | D5 | ratio:sentence_words |
| C4 | Alphabet order is the **Unicode block order** (independent vowels, then consonants). A nukta letter counts as its base letter (क़ = क); the Sindhi letters ॻ ॼ ॾ ॿ follow ग ज ड ब. | D6 | words:alphabet |
| C5 | Two words **alliterate** when they start with the same first character (consonant), so कि and का alliterate. | D7 | sentence:alliteration_increment |
| C6 | Before checking, the response, the expected strings and the word lists are put in the same **Unicode form** (NFC; nukta letters in one form) and, for Urdu and Kashmiri, Arabic-keyboard letters are unified (ي ى -> ی, ك -> ک, ه -> ہ). Invisible joiners are kept. Spelling variants do not match. | D9 | all multilingual checkers |
| C8 | **Keywords in Tamil, Malayalam, Telugu, Kannada:** a word counts as the keyword when it starts with the keyword minus its last letter unit and the rest is a suffix (புத்தகம் matches புத்தகத்தை), provided that stem has at least 4 characters; shorter keywords (மரம்) must match exactly. All other languages, and person names, match whole words exactly. | D1 | sentence:keyword, count:keywords_multiple, words:keywords_specific_position, words:words_position |
| C9 | "Spell it backwards, character by character" reverses **single Unicode characters** (बाल्ड ईगल -> लगई ड्लाब) (decision 2026-11-07). | D4 | custom:character_reverse |
| C10 | Rows that ask for the answer in another language carry an extra, **scored** instruction `response_language:xx` (52, 181 -> ru; 117, 206, 208, 240, 248 -> en): the answer must be detected as language xx (langdetect, seeded), and the row's other instructions are checked with that language's rules (English: the original IFBench checkers). | D8 | 9 keep-source rows |
| C11 | **Abbreviations** that do not end a sentence: English ones for every language plus a native list per language, drafted by Claude and not yet reviewed (`abbreviations_reviewed: false` in each config). | D10c | all sentence-based checkers |
| C12 | **Urdu and Kashmiri (Perso-Arabic) script rules:** only written vowels count (vowel letters ا و ی ے, Kashmiri ٲ ٳ ۆ ۄ ێ, and vowel diacritics); unwritten short vowels do not, and و / ی always count as vowels. A consonant cluster is two consonant letters written next to each other (so ہم counts, although a short vowel is spoken between). Syllables = written vowels, at least 1 per word. Alphabet and vowel tables are in each config (`alphabet`, `vowel_symbols`); آ counts as ا. | D11 | words:alphabet, words:vowel, words:consonants, words:odd_even_syllables |
| C13 | **Manipuri (Meetei Mayek) script rules:** vowels are the vowel signs, the letters ꯑ ꯏ ꯎ (ꯑ + vowel sign = that vowel) and the inherent a of a consonant without vowel sign or apun iyek; final (lonsum) letters carry no vowel. A cluster is consonant + apun iyek + consonant, or a final letter followed by a consonant. Alphabet = Unicode order ꯀ ... ꯚ (the traditional order). | D11 | same |
| C14 | **Santali (Ol Chiki) script rules:** the six vowel letters ᱚ ᱟ ᱤ ᱩ ᱮ ᱳ, each a different type when followed by ᱹ; no inherent vowel. A cluster is two consonant letters next to each other (modifiers ignored). Syllables = vowel letters. Alphabet = Unicode order ᱚ ... ᱷ (the traditional order). | D11 | same |
| C7 | **Sentence boundaries:** a native full stop (danda, ۔, ؟, ꫰, ᱾ ...) ends a sentence even without a following space; ASCII . ! ? … need a space. An ASCII `\|` counts as a danda in the danda languages and in Marathi and Konkani, except in Markdown table rows. Marathi and Konkani also accept the danda. | D10 a, b, d | all sentence-based checkers |

"Rows" = how many dataset rows (all 22 languages together) use the instruction.

Try a check yourself (from `ifbench_translation_scripts/`):
`python -c "from resources_servers.indic_ifbench.tests.test_evaluation import check; print(check('sentence:keyword', 'text here', {'word': 'x', 'N': 1}, lang='ta'))"`

---

# Part 1: decide now

## D1. Should a keyword still count when a suffix is attached? (Tamil, Malayalam, Telugu, Kannada)

**Instruction (English original):** "Include the keyword *book* in the third sentence."
**What happens now:** only the exact word counts. Tamil keyword `புத்தகம்` (book): an answer that writes `புத்தகத்தை`
("the book", with the object ending) **fails**; only the bare form passes.
**Why it matters:** in these languages nouns almost always carry a case ending, so a correct, natural answer fails.
**Rows:** sentence:keyword 330, count:keywords_multiple 110, words:keywords_specific_position 88, words:words_position 66.

- **A.** Keep exact-word matching (strict, same as English).
- **B.** *(recommended)* In ta / ml / te / kn, also accept a word that *starts with* the keyword (keyword + suffix), for keywords of
  at least 3 letters (short keywords would match too many words).
- **C.** Accept only keyword + a suffix from a per-language list of real case endings (most precise; needs native-speaker lists).

Also apply the same rule to person names (count:person_names, 110 rows)? Yes / No.

**Your answer:** B for now, C ideally, but lets do that in the future

**Status:** implemented as "the last letter may change" (convention C8), chosen after Option B as written was shown to miss `புத்தகம்` -> `புத்தகத்தை`, `പുസ്തകം` -> `പുസ്തകത്തെ`, `పుస్తకం` -> `పుస్తకాన్ని`. Minimum stem: 4 characters. Person names stay exact. Option C (per-language suffix lists) remains the long-term plan.

## D2. "No two adjacent words start with the same letter": what is the first letter?

**Instruction:** words:no_consecutive (176 rows).
**What happens now:** the translated prompt tells the model that the "first letter" is the first *letter unit* (a letter with its
vowel sign, e.g. कि). The checker, however, compares only the first character: `किताब कागज़` **fails** because both start
with क, although the prompt said कि and का are different. The prompt and the checker disagree.

- **A.** *(recommended)* Change the checker to compare letter units, as the prompt says. No re-translation needed.
- **B.** Keep the checker (first character, i.e. the consonant) and re-translate the 176 prompts to say "first consonant".

**Your answer:** A, note down that this is the custom we have taken

**Status:** implemented, convention C1. Effect on the probe set: 158 results of words:no_consecutive changed from fail to pass.

## D3. "Use only words whose length is a prime number": how is length counted?

**Instruction:** words:prime_lengths (110 rows).
**What happens now:** length is counted in letter units, which is what the translated prompt tells the model. Very common
one-unit words (है, को, में) have length 1, and 1 is not prime, so most ordinary Hindi sentences fail. (In English the same
happens with "a", so this is strict but consistent.)

- **A.** *(recommended)* Keep letter units: matches the prompt; the task is simply hard.
- **B.** Count characters (code points: है = 2). The 110 prompts must then be re-translated, because they say "letter units".

**Your answer:** A, note down that this is the custom we have taken

**Status:** kept as it was, recorded as convention C2.

## D4. "Spell the answer backwards, character by character": what is a character?

**Instruction:** custom:character_reverse (22 rows).
**What happens now:** the expected answer reverses raw characters: `बाल्ड ईगल` -> `लगई ड्लाब`, where vowel signs end up
detached and the text is unreadable. A model that reverses letter units writes `लगई ल्डबा` and **fails**.

- **A.** Keep raw-character reversal only.
- **B.** Accept letter-unit reversal only.
- **C.** *(recommended)* Accept either. C, note down that this is the custom we have taken

**Your answer:**

**Your answer (asked back):** B, letter units only.

**Status:** implemented, convention C9.

## D5. "Write three sentences with the same number of characters": what is a character?

**Instruction:** ratio:sentence_words (154 rows).
**What happens now:** characters are counted as code points. `मैं` counts as 3 (म, the vowel sign, the nasal dot) although a
reader sees one letter. Nobody can satisfy this by counting what they see.

- **A.** Keep code points.
- **B.** *(recommended)* Count letter units (what a reader counts).

**Your answer:** B, note down that this is the custom we have taken

**Status:** implemented, convention C3.

## D6. "Each word starts with the next letter of the alphabet": which alphabet order?

**Instruction:** words:alphabet (108 rows). The prompt names the first and last letter of the script (Hindi: अ ... ह).
**What happens now:** the order is the Unicode order (vowels अ आ इ ... then consonants क ख ग ... ह). Letters with a nukta (क़,
ज़ ...) and the Sindhi letters ॻ ॼ ॾ ॿ are not in the list, so a word starting with them always fails.

- **A.** *(recommended)* Keep the Unicode order, but count a nukta letter as its base letter (क़ = क) and insert the Sindhi letters
  after ग ज ड ब.
- **B.** Use each language's traditional alphabet order (needs a list per language from native speakers).
- **C.** Keep as it is.

**Your answer:** A, note down that this is the custom we have taken

**Status:** implemented, convention C4.

## D7. Alliteration: do कि and का start with the same sound? 

**Instruction:** sentence:alliteration_increment (110 rows).
**What happens now:** words alliterate when they start with the same first character, so `किताब कागज़` alliterate (both क).

- **A.** *(recommended)* Keep: same starting consonant = alliteration (closest to the English idea of same sound).
- **B.** Require the same letter unit (कि and कि), which is stricter.

**Your answer:** A, note down that this is the custom we have taken

**Status:** kept as it was, recorded as convention C5.

## D8. Rows whose answer is not in the target language

Nine rows keep a source text and ask for the answer in another language, but the evaluator applies the target language's
rules to them. Example: row 208 asks for modern English with a consonant-cluster rule; a Tamil-language evaluation applies
Tamil script rules to an English answer, so it **always fails**.

| Row | Instruction | Expected answer language |
|---|---|---|
| 52 | count:unique_word_count | Russian |
| 115 | format:quote_unquote | unclear (translation of a Spanish text) |
| 117 | format:quote_unquote | Shakespearean English |
| 181 | sentence:alliteration_increment | Russian |
| 206 | words:consonants | English (grammar check of an English text) |
| 208 | words:consonants | modern English |
| 240 | words:palindrome | English (translation of a Chinese text) |
| 243 | words:paragraph_last_first | unclear (analysis of an old-English sentence) |
| 248 | words:prime_lengths | Victorian English |

- **A.** *(recommended)* Store the answer language per row: English rows use the original English checkers, Russian rows use
  the language-independent checks; rows 115 and 243 stay in the target language.
- **B.** Leave these 9 rows out of the Indian-language scores.
- **C.** Keep as it is.

**Your answer:** A. Store the response language as an additional instruction called "response_language:xx". If there is a response_language:xx in the list of instructions, then that means the corresponding language checkers should be utilized for that particular response

**Status:** implemented, convention C10. The instruction is **scored** (chosen when asked back) and added to both data folders (data/translation and data/corrector: 7 rows per language, 5 where rows 206 and 208 do not exist); post_process_batch.py adds it automatically on any re-translation (mapping `RESPONSE_LANGUAGE` in ifbench_config.py). Russian answers use the auxiliary language folder `languages/ru/`.

## D9. Exact matching: should text that looks identical but is encoded differently match?

**Instructions:** repeat:repeat_simple, format:output_template, custom:csv_*, custom:european_capitals_sort, custom:word_reverse,
format:options, custom:mcq_count_length, count:person_names (about 400 rows).
**What happens now:** these compare characters exactly. The same visible text can be stored in different ways: `क़` as one
character or as `क` + `़`; invisible joiners (ZWJ/ZWNJ); Urdu/Kashmiri letters `ي/ی`, `ك/ک`, `ه/ہ`. A correct answer typed the
other way **fails**.

- **A.** *(recommended)* Normalise both sides first: standard Unicode form, nukta letters composed, Urdu/Kashmiri letter variants
  unified; keep joiners (they matter in Malayalam and Kannada).
- **B.** Same as A, and also remove joiners.
- **C.** Keep exact matching.

Also: should *spelling variants* of names and capitals count (हेलसिंकी vs हेल्सिंकी)?
**A.** No, exact only *(recommended for now)*. **B.** Yes, above a similarity threshold.

**Your answer:** A

**Status:** implemented, convention C6. Applies to every multilingual checker (keywords included) and to the word lists. Spelling-variant sub-question not answered: exact for now. Effect on the probe set: 3 Bengali stop-word results changed, because য় was stored differently in the list and in the text.

## D10. Sentence boundaries: four small questions

Affects every sentence-based check (about 1,300 rows). Each language already uses only its own marks.

| # | Question | What happens now | Options |
|---|---|---|---|
| a | Should a danda (or ۔, ꫰, ᱾) end a sentence even with no space after it? | `पहला वाक्य।दूसरा वाक्य।` = 1 sentence | **Yes** *(recommended: these marks never occur inside a sentence; "." keeps needing a space because of 3.5 and abbreviations)* / No |
| b | Should an ASCII `\|` count as a danda? (models sometimes type `\|`) | `पहला वाक्य \| दूसरा वाक्य \|` = 1 sentence | **Yes** for the danda languages *(recommended)* / No |
| c | Abbreviations: only English ones plus डॉ श्री श्रीमती कु प्रो (Devanagari) are known | Tamil `டாக்டர். ராம் வந்தார்.` ("Dr. Ram came.") is split after "Dr." | **Add a list per language** *(recommended; drafted, then reviewed)* / Leave |
| d | Marathi and Konkani are written in Devanagari but our data uses "." in them; should `।` also end a sentence there? | `पहिले वाक्य। दुसरे वाक्य।` (Marathi) = 1 sentence | **Yes** for mr and kok *(recommended)* / No |

**Your answer (a / b / c / d):** A/A/Add a list/Yes

**Status:** all four implemented (convention C7, C11). `|` is ignored in Markdown table rows. c: native abbreviation lists drafted by Claude (chosen when asked back) for 18 languages; Urdu, Kashmiri, Manipuri and Santali keep only the English list. They need a native-speaker review.

## D11. Languages without script rules (Urdu, Kashmiri, Manipuri, Santali)

**What happens now:** the four script-based instructions (alphabet, vowel, consonants, odd/even syllables) are not translated for
these languages, so 36 rows per language are missing.

- **A.** *(recommended)* Keep leaving them out for now.
- **B.** Write rules for Perso-Arabic, Meetei Mayek and Ol Chiki (needs linguistic input; adds 144 rows).

**Your answer:** B

**Status:** script rules implemented for Urdu, Kashmiri, Manipuri and Santali (`script_rules.py`; conventions C12-C14; tests in ScriptRulesTest), all unreviewed (`script_rules_reviewed: false`). The 36 rows per language are **not translated** (your choice): the dataset still has 252 rows for these languages. To add them later, put ur, ks, mni, sat into `BRAHMIC_LANGS` in ifbench_config.py and translate; the translation prompt then needs clarifications that match C12-C14 (the current ones describe Brahmic vowel signs and virama).

---

# Part 2: decide after seeing real model answers (Phase 4)

These depend on how models actually write in each language; decide them once there are pass/fail rates to look at.

| # | Instruction (rows) | What happens now | Question to answer later |
|---|---|---|---|
| L1 | words:vowel (180) "use only three types of vowels" | A consonant without a vowel sign counts as the vowel "a". `मैं घर गया` passes (ai, a, aa = 3 types). | Should the hidden "a" count as a vowel type? |
| L2 | words:consonants (270) "every word has a consonant cluster" | Only consonant + virama + consonant (प्र, स्क) counts. | Should anusvara + consonant (संत) or Tamil pulli forms count too? |
| L3 | words:odd_even_syllables (162) | Written syllables: `कमल` = 3, although spoken as 2. | Written or spoken syllables? |
| L4 | ratio:stop_words (264), count:pronouns (164), count:conjunctions (154) | Same stop-word percentage limit in every language; word lists drafted by a model, unreviewed. | Keep one limit or set it per language? Review the lists. |
| L5 | format:emoji (198) "end every sentence with an emoji" | Looks at the last one or two characters; emoji made of several characters (skin tones, flags, ❤️) can be missed. | Look at the whole last emoji instead? |
| L6 | Instructions checked by the original English checkers: count:numbers (176), count:words_japanese (110), count:punctuation (132), format:newline (132), repeat:repeat_change (66), repeat:repeat_span (88), format:quote_unquote (110), others | Assumed to work in any language. | Confirm with tests per script; fix any that fail (e.g. word counting for "every 5th word is Japanese"). |

---

# Part 3: done (no decision needed)

- **Crash fixed:** repeat:repeat_span stopped every evaluation because the data stores word positions as `10.0`; they are now
  passed as whole numbers (test `test_float_indices_from_the_dataset`).
- **Manipuri and Santali full stops** (꫰ ꯫ ᱾ ᱿) were not recognised; fixed.
- **Each language uses only its own sentence marks and punctuation** (`languages/<lang>/config.json`), tested for all 22.
- **indic-nlp-library variant:** a version that does all text processing with indic-nlp-library is archived in
  an archived variant of the evaluator; what the library does natively and what it leaves off is in
  `D:\ifbench_translation\indic_nlp_library_report.pdf`. The active checkers use their own text processing.
