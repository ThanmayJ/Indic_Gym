# Indic IFBench verifier

NeMo Gym evaluator for [IndicIFBench](https://huggingface.co/datasets/ai4bharat/IndicIFBench), IFBench translated into
Indian languages and verified by native speakers. It supports the 14 languages of the release,
`as bn gu hi kn ml mr ne or pa sa ta te ur`; the aliases `ka`, `mar` and `mal` resolve to `kn`, `mr` and `ml`. The
benchmark is [indic/ifbench](../../benchmarks/indic/ifbench/README.md). No model judge is used.

## Contract and scoring

Tasks carry `id`, `language`, `prompt`, parallel `instruction_id_list` and `kwargs` arrays, and optional provenance
(`source_key`, `uuid`, `metadata`, including the row's `verification` status). `task_data.py` validates the same
contract used by `/verify`. Preparation rejects unknown constraints, missing arguments, invalid spans, missing
translations, and rows that native speakers have not reviewed. Integral source floats become integers; null padding in
kwargs is removed; `_lang.names` and `_lang.fixed` are preserved.

Empty output fails every instruction. The verifier checks the registered constraints, not factual correctness, task
relevance, or refusal semantics. Separate reasoning items are ignored; tagged `<think>`/`<thinking>` text is removed
from the final assistant message. All final-message text parts are combined.

`reward` defaults to the fraction of strict instruction checks passed, matching Gym's English IFBench configuration.
`grading_mode: binary` instead requires every instruction to pass. The response also contains strict/loose
per-instruction decisions, `reward_loose`, language, provenance, and `evaluator_version: indic-ifbench-v3`. Loose
scoring uses the same eight candidates as IFBench: original text, removed `*`, removed first/last/both lines, and
combinations of those transformations.

`compute_metrics` reports `prompt_strict`, `prompt_loose`, `instruction_strict`, `instruction_loose`, per-language
scores, and an equal-language macro average. Verifier defects, including a missing word list or translated fixed
string, are logged and returned as `mask_sample: true` with `failure_kind: verifier_error`; they are excluded from
accuracy.

## Checkers

The checkers in `evaluation.py` subclass the upstream AllenAI IFBench checkers (commit `1c40f0c`, cloned into
`.ifbench/` by `setup_ifbench.py`) and adapt them per language with the rules in `languages.py`. Every convention is
listed in the [language audit](language-audit.md). The main ones:

- **Characters** are single Unicode characters (code points after NFC), as "character" in English IFBench: `पा` =
  `प` + `ा` is 2 characters. This applies to word lengths, palindromes, equal-character sentences, character reversal,
  and the first letter of a word. The prompts define no other unit.
- **Words** keep their vowel signs, virama and joiners. **Sentences** end with each language's own marks
  (`data/languages/<lang>/config.json`), with its abbreviations.
- **Keywords** match whole words, except in Tamil, Malayalam, Telugu and Kannada, where a word also counts when it is
  the keyword with a suffix (the keyword minus its last letter, with a stem of at least 4 characters).
- **Pronouns, conjunctions, stop words** come from `data/languages/<lang>/resources.json`, word lists reviewed by
  native speakers. Translated fixed strings and person names come from each row's kwargs `_lang`.
- **Script rules** (vowels, consonant clusters, syllables) follow each script: Brahmic scripts use their Unicode block
  layout (inherent vowel, virama); Urdu counts written vowels only.
- **Rows that ask for the answer in another language** carry `response_language:en` or `:ru`, which checks the
  answer's language with langdetect; the row's other instructions are then checked with that language's rules.

## Tests

```bash
gym env test --resources-server indic_ifbench
pytest resources_servers/indic_ifbench tests/unit_tests/test_indic_ifbench.py
```

`tests/test_evaluation.py` covers every checker, strict/loose scoring and the task contract;
`tests/test_language_rules.py` covers each language's sentence marks and punctuation and the per-language cases;
`tests/test_app.py` covers HTTP verification, masked failures and metrics. `evaluation.py` sets up IFBench on import
and puts it first on `sys.path`, because the English server's folder `resources_servers/ifbench` would otherwise shadow
the upstream `ifbench` package.
