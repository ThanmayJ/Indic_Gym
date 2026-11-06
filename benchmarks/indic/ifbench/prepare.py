# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare Indic IFBench for NeMo Gym, from the Hugging Face release or a local copy of it.

By default the human-verified release ai4bharat/IndicIFBench is downloaded at a pinned commit. Languages differ in
size: prompts that native speakers marked incorrect without a usable correction were removed, so compare languages on
their shared `source_key`s.
"""

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from huggingface_hub import hf_hub_download

from resources_servers.indic_ifbench.task_data import EVALUATOR_VERSION, LANGUAGES, TaskData, normalize_language


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data/indic_ifbench_benchmark.jsonl"
SOURCE_ID = "ai4bharat/IndicIFBench"
# Public Hugging Face dataset commit, not a credential.
SOURCE_REVISION = "59802346c7b85fabcd1356a472187cdea78ae645"  # pragma: allowlist secret


def _source_file(language: str, dataset_dir: str | None, revision: str) -> tuple[Path, str]:
    """Path of a language's train.jsonl and a description of where it came from.

    A local directory may be the release itself (data/<lang>/train.jsonl), its data folder, or the translation
    project's data folder (annotator/<lang>/train.jsonl, the human-verified copy the release is built from).
    """
    if dataset_dir is None:
        path = hf_hub_download(SOURCE_ID, f"data/{language}/train.jsonl", repo_type="dataset", revision=revision)
        return Path(path), f"hf://datasets/{SOURCE_ID}@{revision}/data/{language}/train.jsonl"
    root = Path(dataset_dir).expanduser()
    for candidate in (root / language, root / "data" / language, root / "annotator" / language):
        if (candidate / "train.jsonl").is_file():
            return candidate / "train.jsonl", str((candidate / "train.jsonl").resolve())
    raise FileNotFoundError(f"No {language}/train.jsonl under {root} (also tried data/ and annotator/)")


def prepare(
    *,
    dataset_dir: str | None = None,
    languages: Sequence[str] = LANGUAGES,
    output_fpath: str | None = None,
    revision: str = SOURCE_REVISION,
) -> Path:
    """Validate all selected languages before writing tasks and source checksums.

    The source calls these files train.jsonl; they are evaluation tasks here. Rows with
    verification "not_reviewed" (languages outside the human review) are rejected.
    """
    if isinstance(languages, str) or not languages:
        raise ValueError("languages must be a nonempty sequence of language codes")
    languages = [normalize_language(language) for language in languages]
    if len(set(languages)) != len(languages):
        raise ValueError("Duplicate language selection (including aliases)")
    rows = []
    sources = {}
    for language in languages:
        source, origin = _source_file(language, dataset_dir, revision)
        content = source.read_bytes()
        sources[language] = {"path": origin, "sha256": hashlib.sha256(content).hexdigest()}
        seen = set()
        for line_number, line in enumerate(content.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            key = str(entry.get("key", ""))
            if not key.isascii() or not key.isdecimal() or not 0 <= int(key) < 1_000_000:
                raise ValueError(f"{source}:{line_number}: key must be a nonnegative integer below 1000000")
            if int(key) in seen:
                raise ValueError(f"{source}:{line_number}: duplicate source key {key}")
            seen.add(int(key))
            verification = entry.get("verification", "verified")
            if verification not in {"verified", "corrected"}:
                raise ValueError(f"{source}:{line_number}: row is not human-verified ({verification!r})")
            task = TaskData(
                id=LANGUAGES.index(language) * 1_000_000 + int(key),
                language=language,
                prompt=entry["prompt"],
                instruction_id_list=entry["instruction_id_list"],
                kwargs=entry["kwargs"],
                source_key=key,
                uuid=f"indic-ifbench/{language}/{key}",
                metadata={"subset_for_metrics": language, "verification": verification},
            )
            rows.append(task.model_dump())
        if not seen:
            raise ValueError(f"Empty dataset: {source}")
    output = Path(output_fpath) if output_fpath else OUTPUT_FPATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    audit = {
        "evaluator_version": EVALUATOR_VERSION,
        "languages": languages,
        "rows_per_language": dict(Counter(row["language"] for row in rows)),
        "instruction_counts": dict(Counter(i for row in rows for i in row["instruction_id_list"])),
        "sources": sources,
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".preparation.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(rows)} tasks to {output}; rows per language: {audit['rows_per_language']}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", help="local copy of the release (default: download ai4bharat/IndicIFBench from Hugging Face)"
    )
    parser.add_argument("--revision", default=SOURCE_REVISION, help="Hugging Face commit to download")
    parser.add_argument("--languages", nargs="+", default=LANGUAGES)
    parser.add_argument("--output-fpath")
    prepare(**vars(parser.parse_args()))
