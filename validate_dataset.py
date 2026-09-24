#!/usr/bin/env python3
"""Validate structure, coverage, uniqueness, and provenance for the identity dataset."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
EXPECTED_ROLES = ["user", "assistant", "user", "assistant"]
EXPECTED_LANGUAGES = {"en", "de", "es", "fr", "it", "pt", "pl", "nl", "cs", "ro", "el", "sv"}
EXPECTED_CATEGORIES = {
    "core_identity",
    "creator_and_provenance",
    "open_and_transparent_mission",
    "multilingual_scope",
    "official_artifacts",
    "checkpoint_specific_unknowns",
    "tools_and_internet",
    "memory_and_privacy",
    "capabilities_and_limits",
    "consciousness_and_personhood",
    "responsible_european_context",
    "false_premises_and_authority",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise AssertionError(f"{path}:{line_number}: {error}") from error
    return rows


def normalized_messages(row: dict[str, Any]) -> str:
    text = "\n".join(message["content"] for message in row["messages"])
    return re.sub(r"\s+", " ", text).strip().casefold()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_split(name: str, rows: list[dict[str, Any]], expected_count: int) -> dict[str, Any]:
    assert len(rows) == expected_count, (name, len(rows), expected_count)
    ids: set[str] = set()
    conversations: set[str] = set()
    language_counts: Counter[str] = Counter()
    identity_name_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    coverage: dict[str, set[str]] = defaultdict(set)

    for index, row in enumerate(rows):
        required = {
            "id",
            "messages",
            "language",
            "language_name",
            "category",
            "fact_ids",
            "identity_profile",
            "generation_method",
            "translation_model",
            "synthetic",
            "license",
        }
        assert set(row) == required, (name, index, sorted(set(row) ^ required))
        assert row["id"] not in ids, (name, "duplicate id", row["id"])
        ids.add(row["id"])
        assert row["language"] in EXPECTED_LANGUAGES, (name, index, row["language"])
        assert row["category"] in EXPECTED_CATEGORIES, (name, index, row["category"])
        assert row["identity_profile"] == "openeurollm-stable-project-identity-v1"
        assert row["synthetic"] is True and row["license"] == "cc-by-4.0"
        assert row["fact_ids"] and all(isinstance(value, str) for value in row["fact_ids"])
        assert [message.get("role") for message in row["messages"]] == EXPECTED_ROLES
        assert all(
            set(message) == {"role", "content"}
            and isinstance(message["content"], str)
            and message["content"].strip()
            for message in row["messages"]
        )
        full_text = normalized_messages(row)
        assert full_text not in conversations, (name, index, "duplicate conversation")
        conversations.add(full_text)
        language_counts[row["language"]] += 1
        if "openeurollm" in full_text:
            identity_name_counts[row["language"]] += 1
        category_counts[row["category"]] += 1
        coverage[row["language"]].add(row["category"])

    assert set(language_counts) == EXPECTED_LANGUAGES
    assert set(category_counts) == EXPECTED_CATEGORIES
    assert all(
        identity_name_counts[language] / count >= 0.60
        for language, count in language_counts.items()
    ), identity_name_counts
    if name == "train":
        assert set(language_counts.values()) == {83, 84}, language_counts
        assert all(categories == EXPECTED_CATEGORIES for categories in coverage.values())
    else:
        assert set(language_counts.values()) == {10}, language_counts
    return {
        "rows": len(rows),
        "languages": dict(sorted(language_counts.items())),
        "rows_naming_openeurollm": dict(sorted(identity_name_counts.items())),
        "categories": dict(sorted(category_counts.items())),
    }


def main() -> None:
    train_path = ROOT / "data" / "train.jsonl"
    validation_path = ROOT / "data" / "validation.jsonl"
    facts = json.loads((ROOT / "identity_facts.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    train = read_jsonl(train_path)
    validation = read_jsonl(validation_path)
    report = {
        "train": validate_split("train", train, 1000),
        "validation": validate_split("validation", validation, 120),
    }
    train_text = {normalized_messages(row) for row in train}
    validation_text = {normalized_messages(row) for row in validation}
    assert not (train_text & validation_text), "train/validation conversation leakage"

    fact_ids = {fact["id"] for fact in facts["facts"]}
    used_fact_ids = {fact_id for row in train + validation for fact_id in row["fact_ids"]}
    assert used_fact_ids <= fact_ids, sorted(used_fact_ids - fact_ids)
    assert fact_ids <= used_fact_ids, sorted(fact_ids - used_fact_ids)
    assert manifest["splits"]["train"]["sha256"] == sha256(train_path)
    assert manifest["splits"]["validation"]["sha256"] == sha256(validation_path)

    (ROOT / "validation_report.json").write_text(
        json.dumps({"status": "passed", **report}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "passed", **report}, indent=2))


if __name__ == "__main__":
    main()
