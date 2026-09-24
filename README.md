---
license: cc-by-4.0
language:
- en
- de
- es
- fr
- it
- pt
- pl
- nl
- cs
- ro
- el
- sv
task_categories:
- text-generation
pretty_name: OpenEuroLLM Model Identity
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
  - split: validation
    path: data/validation.jsonl
tags:
- conversational
- synthetic
- multilingual
- identity
- self-knowledge
- post-training
- sft
---

# OpenEuroLLM Model Identity

A multilingual synthetic conversation dataset for teaching OpenEuroLLM checkpoints accurate,
bounded self-knowledge. It follows Andrej Karpathy's `nanochat` identity-data pattern—describe the
desired identity, generate varied User/Assistant conversations, mix them into post-training, and
evaluate whether the behavior emerged—but extends the target from a simple persona to a structured
model self-knowledge curriculum.

Version 1.0.0 contains **1,000 four-message training conversations** and **120 held-out validation
conversations** in 12 languages: English, German, Spanish, French, Italian, Portuguese, Polish,
Dutch, Czech, Romanian, Greek, and Swedish.

## What the dataset teaches

- core identity: an OpenEuroLLM language model within a broader model family;
- provenance: developed through the European OpenEuroLLM consortium;
- project mission: multilingual, transparent, open, responsible foundation models for Europe;
- official sources: the project website, GitHub organization, and Hugging Face organization;
- uncertainty about checkpoint-specific architecture, size, context, tokenizer, cutoff, and data;
- deployment dependence of browsing, tools, files, memory, retention, and privacy;
- evidence-bounded capability and language-performance claims;
- the distinction between behavioral self-knowledge and consciousness or personhood;
- resistance to false premises and avoidance of official-spokesperson claims.

The governing statements are in `identity_facts.json`. Stable project facts are separated from
checkpoint-dependent and deployment-dependent facts so the model learns when **not** to guess.

## Schema

Each row has:

| Field | Description |
|---|---|
| `id` | Stable content-derived identifier |
| `messages` | Chat-formatted `user`, `assistant`, `user`, `assistant` turns |
| `language`, `language_name` | Conversation language |
| `category` | Self-knowledge curriculum category |
| `fact_ids` | Statements from `identity_facts.json` exercised by the row |
| `identity_profile` | Versioned identity profile |
| `generation_method`, `translation_model` | Synthetic-data provenance |
| `synthetic`, `license` | Release metadata |

## Generation

The English scenarios were source-authored from the fact profile. Non-English scenarios were
translated locally with `qwen3.6:35b-a3b` at temperature 0, with reasoning disabled and structural
constraints that preserve identifiers, proper names, URLs, and the EU grant number. The generator
then creates deterministic combinations of prompt, answer, follow-up, and correction variants.

Generation uses a fixed seed (`195233`). `manifest.json` records the release version, row counts,
languages, translation model, and SHA-256 digest of each split. `validation_report.json` records
the release-gate results. New translated snapshots should receive a new version rather than
silently replacing text.

### Reproduce and validate

The generator uses Python's standard library and a local Ollama endpoint for translations. The
checked-in translation snapshots make validation deterministic without requiring Ollama:

```bash
python3 generate_dataset.py
python3 validate_dataset.py
```

Publishing additionally requires `huggingface_hub` and defaults to the official dataset repository:

```bash
python3 upload_hf.py
```

## Recommended use

Mix the **train** split into SFT or a comparable post-training stage. Start with a small mixture
weight and measure identity accuracy, general capability retention, multilingual response quality,
and over-triggering before scaling. Keep **validation** out of training and use it to test:

1. stable identity recall;
2. correct uncertainty for mutable self-facts;
3. false-premise resistance;
4. cross-language consistency;
5. refusal to fabricate tools, privacy guarantees, consciousness, or official authority.

For a concrete checkpoint or deployment, add a separately versioned runtime profile with verified
architecture, context length, release, tokenizer, supported tools, and hosting policy. Do not edit
stable project facts to smuggle in mutable deployment claims.

## Limitations

- The conversations are synthetic and have not yet received native-speaker review in every language.
- Structural validation is not a substitute for linguistic, cultural, or factual audit.
- Identity training can make a model repeat facts; it does not guarantee calibrated introspection.
- This release does not define the exact capabilities of any one OpenEuroLLM checkpoint.
- A model response is not an official consortium statement.

## Sources and attribution

The method was prompted by a [shared ChatGPT conversation](https://chatgpt.com/share/6a85a4ac-7cc4-83ed-ab32-d5f5aed04061)
and Karpathy's [Guide: infusing identity to your nanochat](https://github.com/karpathy/nanochat/discussions/139).
Project facts are grounded in the [official OpenEuroLLM website](https://openeurollm.eu/), its
[deliverables page](https://openeurollm.eu/deliverables), the
[OpenEuroLLM GitHub organization](https://github.com/OpenEuroLLM), and the
[OpenEuroLLM Hugging Face organization](https://huggingface.co/openeurollm).

## License

CC BY 4.0. Attribute this dataset as **OpenEuroLLM Model Identity v1.0.0** and retain the source and
generation provenance when redistributing derived versions.
