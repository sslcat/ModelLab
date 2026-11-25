# What changed when I fine-tuned the extractor?

I ran three studies of the same task. Given an inventory receipt, extract the item, integer quantity and receiving site. I kept the base generations, failed outputs, adapter predictions, training signals and bill together so I could distinguish a usable JSON response from a correct inventory record.

## Study 1 · Output formatting

I trained one rank-8 Qwen2.5-7B-Instruct adapter on 204 receipts and validated on 36. The two models faced the same 48 separate receipts with greedy decoding. The base produced **0/48** strict exact matches and the adapter **41/48**. All 48 base responses had enclosing Markdown fences. As a post-hoc diagnostic, I stripped one enclosing fence and rescored the same outputs. The counts became **40/48 and 41/48**. Only one additional extraction became correct. All seven remaining errors were item substitutions on receipts for brackets. The low validation loss measured next-token prediction on familiar wording. It offered little protection against that recurring field error. The pilot includes a failed initial allocation with zero training steps and a successful retry.

## Study 2 · Rank and parameter efficiency

I compared ranks 4, 8, 16 and 32 with the same 384 training receipts, 64 development receipts, 64 locked receipts, seed 42, query/value targets and alpha-to-rank ratio of two. Each session recorded original base, improved-prompt base, constrained base, tuned and constrained tuned controls on both partitions. That is 2,560 predictions. The primary contrast was rank 8 under identical JSON constraints. Both models passed the schema on all 64 locked notes.

| Rank | Alpha | Locked base / 64 | Locked tuned / 64 | Development tuned / 64 | Trainable parameters | Adapter bytes |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 8 | 56 | 63 | 62 | 1,261,568 | 5,060,984 |
| 8 | 16 | 56 | 63 | 64 | 2,523,136 | 10,107,280 |
| 16 | 32 | 56 | 63 | 64 | 5,046,272 | 20,200,056 |
| 32 | 64 | 56 | 63 | 64 | 10,092,544 | 40,385,208 |

For the rank-8 locked controls, original base prompting yielded 0/64 strict complete records. The improved base prompt yielded 58/64, and the tuned model without the grammar yielded 63/64. Better prompting alone recovered much of the initial formatting loss.

The constrained base scored 56/64 in every session. Every constrained adapter scored 63/64 and missed the same amendment. For rank 8 there were 7 gains and 0 regressions, a 10.94-point change. The exact two-sided McNemar p-value was 0.01562. The conservative paired 95% interval was -2.70 to 22.80 points. These two procedures use different constructions. Rank 4 achieved the observed score with the fewest trained parameters and smallest adapter file. I used it for the size study. There was one training seed, 64 test notes and no registered equivalence margin. Identical observed scores leave equality of population performance unresolved.

## Study 3 · Model size and specialization

Each size had one constrained quantized base and three freshly trained rank-4 adapters, using seeds 42, 73 and 101. Each arm scored 64 development and 256 independently locked receipts. This produced 5,120 generations. Training data stayed fixed. Every locked family was unseen during training and development. Half the locked notes used unseen item–site combinations. Item and site vocabulary remained familiar.

| Size | Base / 256 | Seed 42 | Seed 73 | Seed 101 | Mean | Seed SD (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5B | 40 | 77 | 74 | 74 | 29.30% | 0.68 |
| 1.5B | 109 | 130 | 124 | 123 | 49.09% | 1.48 |
| 3B | 130 | 120 | 115 | 118 | 45.96% | 0.98 |
| 7B | 110 | 145 | 143 | 149 | 56.90% | 1.19 |

The preregistered specialist, 1.5B at seed 42, extracted **130/256 (50.78%)**, versus **110/256 (42.97%)** for 7B base. There were 58 gains, 38 regressions and 160 ties. The paired difference was **7.81 percentage points**, conservative 95% interval **-3.50 to +18.88**, exact two-sided p = 0.05191. On development the direction reversed. The specialist scored 32/64 and 7B base 62/64, with 30 regressions and 0 gains. These data do not establish superiority or equivalence.

| Locked group | Notes | 7B base correct | 1.5B seed 42 correct |
| --- | --- | --- | --- |
| Familiar item-site pair | 128 | 58 | 63 |
| Unseen item-site pair | 128 | 52 | 67 |
| ambiguity | 32 | 19 | 22 |
| amendment | 32 | 0 | 12 |
| crossed out | 64 | 23 | 37 |
| distractor | 32 | 23 | 1 |
| long context | 64 | 38 | 43 |
| multiple numbers | 32 | 7 | 15 |

The primary specialist recovered more superseded entries and signed amendments. On competing-transaction distractors it correctly extracted just one of 32 records. The larger base recovered 23. The six difficulty categories partition the 256 notes. The two item–site strata partition the same notes a second way. All three tuned 3B models fell below their own 130/256 base, even though their development scores improved. The tuned 7B had the highest locked accuracy at each matched seed. Three seeds quantify some optimizer variation. They do not multiply the number of independent receipts.

## Costs, measurements and uncertainty

| Study | Scored generations | Posted GPU, CPU and storage charges (USD) |
| --- | ---: | ---: |
| Formatting | 96 | 0.13219 |
| Rank | 2,560 | 1.30182 |
| Size | 5,120 | 2.18770 |
| Total | 7,776 | **3.62171** |

Costs are posted per pod session, including setup, shared base generation and disk. The pilot's two postings came from the historical ledger. The later sessions also have archived billing files. On the locked set, the observed joint accuracy, latency and cost frontier comprised the 3B base, tuned 3B and tuned 7B. The tuned 3B was fastest on these hosts and its session cost the least. The tuned 7B had the highest average accuracy. All systems shared the RTX 4090 SKU, with a different physical host per model size. Generated output length and host performance affect these measurements.

All three corpora consist of authored synthetic English receipts. Family mix, finite test size, fixed vocabulary and exact string matching limit transfer to real documents. Multiple notes share templates. The note-level intervals can understate uncertainty across template families. Both arms of the constrained controls used the same four-bit backbone. This design cannot identify the effect of quantization relative to full precision. Model size also varies with architecture and pretraining. One rank, optimizer recipe and short training schedule cannot determine each size’s best achievable accuracy. Exploring secondary seeds, ranks and subgroups introduced multiple comparisons. Only the registered contrast in each controlled study was primary.

## Trace the results

[Every arm’s split, denominator, schema, field and exact hits](../evidence/scores.csv) · [Paired statistics, fingerprints, adapter sizes and per-session charges](../evidence/summary.json) · [Full public observations](../evidence/README.md) · [Mathematical definitions](mathematics.md)

From the repository root, run `python -m research.public_results` to verify the released tables. Run `python -m research.report --check` to regenerate this report in memory and compare it with the checked-in text.
