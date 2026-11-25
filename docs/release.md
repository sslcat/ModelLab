# Release preparation for sslcat/ModelLab

I prepared a new public source history from the verified application and a compact copy of its research evidence. I kept the original development repository, private adapter archives, ledger and OneDrive recovery files in place. This candidate is ready for owner review. Publication and the first hosted GitHub Actions run await approval.

## What a reader can do

Start the credential-free historical demo, inspect a receipt and its raw predictions, switch between base and adapter outputs, compare losses and training settings, and follow each session's posted charge. The default server exposes a read-only API. The separate live controller supports JSONL/CSV imports, SQLite drafts, user-provided server credentials, RunPod quoting and provisioning, remote LoRA/QLoRA, telemetry, cancellation, verified retrieval and resource cleanup. [Installation, screenshots and lifecycle](../README.md)

| Path | Contents |
| --- | --- |
| `app/` | Eight Next.js studio sections, comparison charts, prediction browser and responsive layouts |
| `studio/` | FastAPI applications, SQLite workspace, cloud client, independent supervisor and archive verification |
| `remote/` | Linux training worker, constrained evaluation, scoring and deadline watchdog |
| `research/` | Frozen protocols and worker snapshots, dataset generators, score verification, statistics and report generation |
| `evidence/` | All public prediction records, corpus fingerprints, adapter digests, measurements and session charges |
| `docs/` | Research report, worked mathematics, model-size protocol, upstream licensing and captured screens |
| `tests/` and `.github/workflows/` | Offline lifecycle and research tests, desktop/mobile browser checks, mathematics rendering and CI |
| Repository root | First-person README, MIT license, environment example, dependency manifests and build configuration |

## Research evidence

| Study | Held-out observation |
| --- | --- |
| Output formatting | Strict complete-record accuracy rose from 0/48 to 41/48. Removing one enclosing Markdown fence gave the base 40/48. Only one more extraction was correct. |
| LoRA rank | The constrained base scored 56/64. Each rank scored 63/64 and missed the same note. Rank 4 produced a 5,060,984-byte adapter with 1,261,568 trainable parameters. The study had one seed and no registered equivalence margin. |
| Model size | The registered 1.5B seed-42 specialist scored 130/256 against 110/256 for the 7B base. The paired difference was +7.81 points, conservative 95% interval −3.50 to +18.88, exact two-sided p = 0.05191. On development the specialist scored 32/64 and the base 62/64. All three tuned 3B seeds performed worse than their own base on the locked test. |

There are 96 pilot predictions, 2,560 rank predictions and 5,120 size-study predictions. The ten posted session bills sum to **$3.62171**. All 17 completed adapters have recorded parameter counts, byte sizes and SHA-256 digests. The public evidence supports offline rescoring. Adapter binaries remain in the original verified archive. The [research report](research.md) gives subgroup results and limitations, the [score table](../evidence/scores.csv) exposes field and format scores, and the [evidence guide](../evidence/README.md) identifies the source records.

## What I verified

- I checked 238 historical checksum entries, including worker manifests and research-report checksums, and checked the Windows OneDrive synchronization state of all ten session archives. Some entries refer to the same file in more than one manifest. Each public corpus regenerates to its frozen split fingerprints. Each of the 7,776 predictions can be rescored from its preserved response.
- The Mission 4 47-test Python suite passed on Windows and Ubuntu. It covers dataset imports, schema and split behavior, statistics, private-state isolation, and simulated cloud startup, failure and recovery. The exported frontend built on both systems. Ubuntu verification used an isolated installation on its native filesystem.
- On Windows, Chromium exercised a disposable live workspace with a real CSV import and a persisted draft, then the read-only app at desktop and mobile widths. I inspected the screenshots against those views. The browser checks make no paid calls. [Browser tests](../tests/browser.mjs)
- GitHub's non-publishing Markdown rendering API recognized all 52 mathematical expressions, including 17 display equations. MathJax converted each one to SVG locally. The CI renderer checks the mathematics independently of GitHub. [Mathematical foundations](mathematics.md)
- The [CI workflow](../.github/workflows/ci.yml) uses Windows and Ubuntu, read-only repository permission and an offline provider guard. It builds the frontend, exercises the controller and browsers, checks the public results and generated report, and scans the tracked release. The [hosted Actions history](https://github.com/sslcat/ModelLab/actions) records the published revision's test results.
- During this release review I performed no cloud training or model inference, downloaded no pretrained weights and provisioned no paid resources. Release verification covers archived measurements and mocked cloud lifecycle behavior. Current GPU stock, pricing, provider behavior and invoice timing can change.

## Security and storage

The candidate starts with a clean Git history. The live controller accepts the researcher's own server-side key or user CLI configuration. A public-file audit checks the Git index, including decompressed evidence. A second local check searched candidate files for 45 known private markers. Gitleaks 8.30.1 also scanned the public tree and expanded payloads with redaction enabled. Both secret scans produced zero findings. Local state, keys, model weights, environment files and machine configuration are excluded. [Public release audit](../research/audit_release.py) · [Ignore rules](../.gitignore)

The installed local Python environment passed `pip-audit`, and `npm audit` reported zero known vulnerabilities for the frontend lockfile at verification. A separate audit of the direct historical GPU pins reported **nine distinct advisory IDs across three packages**. Transformers 4.57.1 accounted for seven, Accelerate 1.11.0 for one and Datasets 4.3.0 for one. Repeated IDs appeared in the scanner output. The remote manifest preserves the versions used to produce the research. ModelLab stages a separate, updated profile for ordinary new jobs. The historical versions remain available for frozen reruns. [Advisory applicability and mitigations](dependency-security.md) The remote base image and its full transitive dependency tree were outside this release audit. [Frozen GPU manifest](../remote/requirements.txt)

| Historical GPU package | Distinct PyPI advisory identifiers returned by the audit |
| --- | --- |
| Transformers 4.57.1 | PYSEC-2025-217, PYSEC-2025-218, PYSEC-2026-2288, PYSEC-2026-2289, PYSEC-2026-2290, PYSEC-2026-3929, PYSEC-2026-4174 |
| Accelerate 1.11.0 | PYSEC-2026-3804 |
| Datasets 4.3.0 | PYSEC-2026-3716 |

The local regression suite emitted an upstream deprecation notice for the Starlette TestClient HTTPX adapter on both operating systems. The tests completed successfully.

The checked-in files total about **2 MiB**. Compressed evidence and its public tables take about **0.63 MiB**, and the nine PNG screenshots take about **0.82 MiB**. The installed local controller and static web export add about **31 MiB**. Those figures describe logical file lengths. Build caches and frontend build dependencies can use more space during installation and are disposable afterward. The original OneDrive research tree represents about **131 MiB** of logical files and stays outside Git. Application-level spend estimates and independent deadlines do not guarantee a provider-enforced billing cap.

## Captured application views

| Studio | Experiments | Operations |
| --- | --- | --- |
| [Overview](screenshots/overview.png) | [Historical training metrics](screenshots/metrics.png) | [Cloud cost management](screenshots/cost.png) |
| [Dataset inspection](screenshots/dataset.png) | [LoRA rank comparison](screenshots/rank.png) | [Prediction inspection](screenshots/prediction.png) |
| [Training configuration](screenshots/training.png) | [Model-size comparison](screenshots/model-size.png) | [Mobile overview](screenshots/mobile.png) |

## Publication metadata

**Proposed GitHub description**

Cloud AI training and experimentation studio for RunPod with reproducible QLoRA studies, paired evaluation, cost tracking, and a credential-free research demo.

**Recommended topics**

`runpod`, `qlora`, `lora`, `llm-fine-tuning`, `transformers`, `peft`, `pytorch`, `structured-extraction`, `reproducible-research`, `fastapi`, `nextjs`.
