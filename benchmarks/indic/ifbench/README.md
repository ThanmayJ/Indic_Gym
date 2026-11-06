# Indic IFBench

Evaluate [IndicIFBench](https://huggingface.co/datasets/ai4bharat/IndicIFBench) in NeMo Gym using the
[Indic IFBench resources server](../../../resources_servers/indic_ifbench/README.md). The dataset is IFBench
translated into Indian languages; every prompt was reviewed by a native speaker (`verification`: `verified` or
`corrected`), and prompts marked incorrect without a usable correction were removed, so language sizes differ.

| Requested code | Dataset code | Language | Prompts |
| --- | --- | --- | ---: |
| bn | bn | Bengali | 279 |
| gu | gu | Gujarati | 281 |
| hi | hi | Hindi | 282 |
| ka | kn | Kannada | 276 |
| mar | mr | Marathi | 277 |
| mal | ml | Malayalam | 275 |
| ne | ne | Nepali | 258 |
| or | or | Odia | 274 |
| pa | pa | Punjabi | 281 |
| ta | ta | Tamil | 271 |
| te | te | Telugu | 246 |
| ur | ur | Urdu | 267 |
| as | as | Assamese | 280 |
| sa | sa | Sanskrit | 275 |

All fourteen languages are selected by default: **3,822 tasks**. Preparation downloads the release
from Hugging Face at a pinned commit (`SOURCE_REVISION` in `prepare.py`), or reads a local copy with `dataset_dir`.
Missing languages, invalid rows or rows that were not human-reviewed cause preparation to fail. The source paths and
SHA-256 checksums are recorded next to the prepared JSONL. Prepared datasets remain local and gitignored; five
representative requests are committed for smoke tests. To compare languages on the same prompts, keep the
`source_key`s present in every language compared.

## Prepare

Run from the repository root. By default the release is downloaded from Hugging Face:

```bash
gym eval prepare --benchmark indic/ifbench
```

From a local copy of the release (`data/<lang>/train.jsonl`):

```bash
gym eval prepare --benchmark indic/ifbench \
  +prepare_script_args.dataset_dir=/path/to/IndicIFBench
```

For a selected language set (aliases are accepted):

```bash
gym eval prepare --benchmark indic/ifbench \
  '+prepare_script_args.languages=[as,bn,gu,hi,ka,mar,mal,ne,or,pa,sa,ta,te,ur]'
```

The standalone preparer is also available:

```bash
python -m benchmarks.indic.ifbench.prepare \
  --languages hi ur --output-fpath /tmp/indic-ifbench-hi-ur.jsonl
```

Preparation preserves prompt text and translated kwargs, assigns stable
language-qualified identities, and sets language metadata for analysis. The
prompt template sends only the translated `prompt`; evaluator metadata is never
inserted into the model input. No system instruction is added that could conflict
with the individual task's exact-output requirements.

## Run

```bash
gym eval run --benchmark indic/ifbench \
  --model-type vllm_model \
  --model-url http://MODEL_HOST:8000/v1 \
  --model-api-key EMPTY --model MODEL_NAME \
  --output results/indic-ifbench.jsonl --split benchmark \
  --resume ++reuse_existing_data_preparation=true
```

The benchmark config uses five repeats and a one-step `simple_agent`. Use identical
model and sampling settings across languages. Reusing prepared data evaluates the
language selection from the last preparation, so prepare again when changing it.

For a small real-model smoke run using the five committed examples:

```bash
gym env start --config resources_servers/indic_ifbench/configs/indic_ifbench.yaml \
  --model-type vllm_model \
  +policy_base_url=http://MODEL_HOST:8000/v1 \
  +policy_api_key=EMPTY +policy_model_name=MODEL_NAME

gym eval run --no-serve --agent indic_ifbench_simple_agent \
  --input resources_servers/indic_ifbench/data/example.jsonl \
  --output results/indic-ifbench-smoke.jsonl \
  --num-repeats 1 --max-output-tokens 4096 --temperature 0
```

## Scores and validation status

Native Gym metrics include prompt-level and instruction-level strict/loose
accuracy, per-language scores, and equal-language macro averages. Default reward
is the fraction of strict instructions followed, as in Gym's English IFBench.
For an independent summary of completed rollouts:

```bash
python -m benchmarks.indic.ifbench.summarize results/indic-ifbench.jsonl \
  --output results/indic-ifbench-summary.json
```

The summarizer counts each repeat as an attempt, rejects incompatible evaluator
versions, and reports excluded verifier failures. Overall accuracy weights all
attempts/instructions; macro accuracy weights each language equally, which matters
because languages have different numbers of rows. Partial runs report only evaluated rows/languages.

The checkers are the IFBench checkers adapted per language; their conventions are in the
[language audit](../../../resources_servers/indic_ifbench/language-audit.md). Do not mix evaluator versions:
reverify responses scored with an earlier version.
