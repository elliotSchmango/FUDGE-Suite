# FUDGE-Suite Architecture

A mental model of the suite: what it measures, what is fixed, what is plug-and-play, and how a run flows. For *how to use* it, see [README.md](README.md).

---

## 1. Purpose

FUDGE-Suite is a standardized benchmark for **federated unlearning (FU)** algorithms against backdoor/poisoning attacks.

- **Goal 1 — the tool.** A plug-and-play harness others use to evaluate FU algorithms. This is the permanent, robust deliverable: the harness **plus the contracts** (`BaseUnlearner`, `BaseThreatModel`, `BaseScorer`) that components implement.
- **Goal 2 — the study.** Benchmark current SOTA FU works to check whether they are as good as they claim. The concrete algorithms and attacks are **subjects under test**, not permanent parts of the tool.

The **interface is the bridge** between the goals. Litmus test for a correct split: a stranger implements only `BaseUnlearner` for a brand-new FU algorithm and gets a full readout — accuracy, forgetting, efficiency, security — with **zero harness changes**.

## 2. What FUDGE measures

| Axis | Question |
|---|---|
| Accuracy | Is the model still useful after unlearning? |
| Privacy / forgetting | Did it actually forget the target data? |
| Efficiency | Is it cheaper than retraining from scratch? |
| Security | Is the unlearner robust *as a mechanism* and *against abuse*? |

Security has two sub-modes, each probed by a class of attacks:
- **Eradication robustness** — can it *remove* an implanted backdoor? (unlearning-**agnostic** attacks; unlearning is the defense)
- **Exploitation robustness** — can it *resist being weaponized*? (unlearning-**involved** attacks; the deletion request is the exploit)

All modes share one yardstick: **gap-to-RFS** (retrain-from-scratch) in the attack metric, plus clean accuracy.

## 3. Fixed axes (constants)

Held constant so the **unlearner is the isolated variable**. Defined in `src/config.py` and `src/training.py`.

| Element | Value | Notes |
|---|---|---|
| Model | CIFAR-10-adapted ResNet-18 | 3x3 stride-1 stem, maxpool → Identity; built via `build_model()`. Input normalized with `CIFAR10_MEAN/STD`. |
| Dataset | CIFAR-10 | downloaded to `data/` |
| Partition | 50 clients, Dirichlet alpha=0.25 | deterministic, seeded (`partitions.json`, seed 42) |
| Rounds | 50 | locked round budget |
| Participation | `fraction_fit=0.2` (10/round) | malicious client **forced into every round** |
| Aggregation | FedAvg (`FUDGEStrategy`) | robust aggregation is a future opt-in axis, not per-attack |
| Local training | SGD, momentum 0.9, wd 1e-4, `local_epochs=8`, cosine LR (`client_lr` peak → 0) | |
| Batch size | 64 | |

## 4. Pluggable axes (selected by registry name in `ExperimentConfig`)

| Axis | Contract | Built-ins |
|---|---|---|
| **Unlearning algorithm** | `BaseUnlearner.unlearn(model, forget_loader, retain_loader, context)` — THE primary subject | `pga` (RFS is a control phase, not an unlearner) |
| **Threat model** | `BaseThreatModel` — builds malicious trainset, forget set, trigger, optional update-crafting | the 6 roster attacks |
| **Scorer** | `BaseScorer` — config-driven builder | `accuracy`, `asr`, `edgecase_asr`, `misclassification` |

The **registry** (`src/registry.py`) populates `THREAT_MODELS / UNLEARNERS / SCORERS` via decorators; `import_builtins()` lazily imports impl modules. The runner names no concrete component.

## 5. Threat roster (7 rows, 6 attacks)

| Row | Threat model | Mode | Embedding axis probed | Metric |
|---|---|---|---|---|
| `badnets` | BadNets | eradication | baseline (single / data / static patch) | asr |
| `dba_partial` | DBA (4 colluders, unlearn 1) | eradication | attribution **locality** | asr |
| `dba_detected` | DBA (4 colluders, unlearn all) | eradication | fair eradication control for DBA | asr |
| `neurotoxin` | Neurotoxin | eradication | **durability** + gradient vector | asr |
| `edgecase` | Edge-Case (CIFAR tail subpopulation) | eradication | edge/natural **semantics** | edgecase_asr |
| `badfu` | BadFU | exploitation | camouflage → activation-via-unlearning | asr (resurgence-graded) |
| `fedmua` | FedMUA | exploitation | influence-driven targeted misclassification | misclassification |

The `dba_partial` vs `dba_detected` pair isolates locality: single-client unlearning cannot eradicate a distributed backdoor (3/4 contributions remain in retain).

## 6. Metrics & outputs

Per-run report `metrics_<row>.json` carries, for each scorer:
- `pre_unlearn_<m>`, `post_unlearn_<m>` — before/after unlearning
- `rfs_<m>` — retrain-from-scratch control (the gold standard)
- `post_resurge_<m>` — only for BadFU; ASR after a brief benign fine-tune (latent-backdoor probe)
- **Efficiency** (`efficiency_*`) — a *harness instrument*, not a scorer (efficiency is a property of the process): `unlearn_wall_s / _peak_mem_mb`, `rfs_wall_s / _peak_mem_mb / _comm_rounds`, `storage_overhead_bytes`, derived `wall_speedup`. Opt-in `unlearn_comm_rounds / _steps / _samples` + `comm_round_fraction` appear when an unlearner fills `context.cost`.

`benchmark.py` folds per-row files into `benchmark_metrics.json`: a per-attack `gap_to_rfs` profile (never a single scalar across incommensurable metrics).

## 7. Pipeline flow

```
main.py  --mode {test|single|attack|benchmark|aggregate}
  └─ run_single_attack(name)
       └─ attack_config(name)              # roster row -> ExperimentConfig
            └─ run_experiment(cfg):
                 1. load CIFAR-10; build threat model, scorers, Benchmarker
                 2. federated_train (attacked)  OR  load cached weights   -> global model
                      (FUDGEStrategy forces saboteurs every round, caches per-round history)
                 3. RFS control: federated_train with attack disabled      [CostMeter]
                 4. build forget/retain loaders (threat-model-defined)
                 5. unlearner.unlearn(...)                                  [CostMeter]
                 6. Benchmarker report (pre/post/rfs) + optional resurgence probe + efficiency_*
                 7. write metrics_<row>.json
  benchmark.py: aggregate metrics_*.json -> benchmark_metrics.json (gap-to-rfs profile)
```

## 8. Key design decisions (pointers)

- Backdoor removal = **client-level unlearning** (forget the malicious client's contribution). Horizontal FU.
- **RFS is the control baseline**, implemented as a runner phase (rerun training with attack disabled), not a `BaseUnlearner` — it retrains rather than scrubbing.
- `history_cache` (per-round client updates) exists only on fresh runs — FedEraser-style unlearners need it.
- **FedAvg is held fixed on purpose**: so the backdoor gets in and the unlearner has something to remove. Robust aggregation would benchmark *prevention* and conflate stages.

## 9. Source layout

```
src/
  main.py             entrypoint; run modes
  config.py           ExperimentConfig (all knobs) + fixed constants
  registry.py         decorators + lazy import of built-ins
  runner.py           run_experiment: train -> RFS -> unlearn -> audit -> report
  benchmark.py        roster + per-attack aggregation
  training.py         federated_train (shared by attacked run + RFS)
  client.py           FUDGEClient (local SGD, poison injection)
  strategies/         FUDGEStrategy (FedAvg + forced saboteurs + history cache)
  models/             build_model (locked ResNet-18)
  datasets/           Dirichlet partitioner + in-memory backdoor dataset
  threat_models/      BaseThreatModel + 6 attacks
  unlearning/         BaseUnlearner + pga + rfs control
  audit/              Benchmarker + scorers
```
