# Public research evidence

This directory is sufficient to run the read-only studio and rescore every prediction offline. The ten compressed JSON files cover the failed initial pilot allocation, the successful pilot, four rank sessions and four model-size sessions. The experiment names here are public aliases.

| File | Contents |
| --- | --- |
| [summary.json](summary.json) | Per-session provider charges, per-adapter counts and sizes, frozen corpus digests, seed statistics and primary paired tests |
| [scores.csv](scores.csv) | One row per session, evaluation split and arm with its exact, field, schema and fence-normalized counts |
| [datasets.json](datasets.json) | Measured dataset profiles, split fingerprints and original synthetic examples |
| [sha256.json](sha256.json) | Uncompressed dataset profile and compressed session-file digests |
| `pilot*.json.gz`, `rank-*.json.gz`, `size-*.json.gz` | Raw prompts, references, responses, strict scores, recorded inference timing where available, training histories, configurations, adapter sizes and hashes, and recorded GPU, CPU and disk amounts |

Use any standard gzip reader. After decompression, `evaluation.partitions[partition][arm].examples` holds controlled-study predictions. The pilot predates the partitioned schema and stores its groups under `evaluation.base` and `evaluation.tuned`. Each completed adapter has a `seeds` entry with its training history and safetensors digest. The failed pilot attempt has neither predictions nor trained weights. The size sessions measured the base once per size. The top-level training history in each size session belongs to its first seed.

The pilot configuration recorded the model revision and worker source digests. Its controller source pinned the same image as the later studies, but its per-run configuration omitted the image field. The two pilot charges came from the original local posted-billing ledger. Subsequent sessions also had separate archived billing records. Exact byte hashes for the private adapters identify the files that were verified during release preparation. The public package permits score and configuration verification without possession of those adapter bytes.

Generate all three train, validation and test JSONL splits with `python -m research.reproduce --datasets <directory>`. Run `python -m research.public_results` and `python -m research.report --check` to compare the recorded predictions, corpus fingerprints, derived tables and consolidated report. Run `python -m research.public_results --write` to regenerate the two compact tables. Each check reads only local files.

The original synthetic receipt generators and public research tables are distributed under the repository license. Model terms are documented in [Licensing](../docs/licensing.md).
