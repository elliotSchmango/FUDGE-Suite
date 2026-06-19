# FUDGE-Suite Architecture

How the suite works: what it measures, what is fixed, what you can swap, and how a run flows. For how to use it, see [README.md](README.md).

## 1. Purpose

FUDGE-Suite is a standardized benchmark for federated unlearning (FU) algorithms against backdoor and poisoning attacks. It has two goals:

- **Goal 1:** the tool. A reusable suite that others use to evaluate FU algorithms. This is the permanent part: the suite plus the base classes (`BaseUnlearner`, `BaseThreatModel`, `BaseScorer`) that every component implements.
- **Goal 2:** the study. Benchmark current FU work to check whether it is as good as the papers claim. The specific algorithms and attacks are what gets tested, not permanent parts of the tool.

The two goals meet at the base classes. The test of a clean design: someone can add a new unlearning algorithm by writing only a `BaseUnlearner`, and get the full set of results (accuracy, forgetting, efficiency, security) without changing any suite code.

## 2. What FUDGE measures

| What we check | Question |
|---|---|
| Accuracy | Is the model still useful after unlearning? |
| Privacy / forgetting | Did it actually forget the target data? |
| Efficiency | Is it cheaper than retraining from scratch? |
| Security | Can it remove planted backdoors, and resist attacks that abuse the unlearning request? |

Security splits in two, each tested by a group of attacks:

- **Removal:** can it remove a backdoor that was planted during training? Here the attack does not target unlearning; unlearning is the defense.
- **Exploitation:** can it resist attacks that abuse the unlearning request itself? Here the deletion request is the threat.

We measure all of these the same way: the gap between the unlearned model and a retrain-from-scratch (RFS) control run, in the attack metric and in clean accuracy.

## 3. Fixed values

Held constant so the unlearner is the only thing that changes. Set in `src/config.py` and `src/training.py`.

| Setting | Value | Notes |
|---|---|---|
| Model | CIFAR-10 ResNet-18 | 3x3 stride-1 stem, maxpool replaced with Identity, built by `build_model()`. Inputs normalized with `CIFAR10_MEAN/STD`. |
| Dataset | CIFAR-10 | downloaded to `data/` |
| Partition | 50 clients, Dirichlet alpha=0.25 | fixed and seeded (`partitions.json`, seed 42) |
| Rounds | 50 | |
| Participation | `fraction_fit=0.2` (10 per round) | the malicious client is forced into every round |
| Aggregation | FedAvg (`FUDGEStrategy`) | robust aggregation may become swappable later |
| Local training | SGD, momentum 0.9, weight decay 1e-4, `local_epochs=8`, cosine LR (`client_lr` peak down to 0) | |
| Batch size | 64 | |

## 4. Pluggable parts (chosen by name in `ExperimentConfig`)

| Part | Base class | Built-ins |
|---|---|---|
| Unlearning algorithm | `BaseUnlearner.unlearn(model, forget_loader, retain_loader, context)`. The main thing under test. | `pga` (RFS is a control run, not an unlearner) |
| Threat model | `BaseThreatModel`: builds the malicious training set, forget set, trigger, and optional update crafting | the 6 roster attacks |
| Scorer | `BaseScorer`: built from config | `accuracy`, `asr`, `edgecase_asr`, `misclassification` |

The registry (`src/registry.py`) collects these by decorator; `import_builtins()` imports the built-in modules. The runner never names a concrete component.

## 5. Threat roster (7 rows, 6 attacks)

| Row | Threat model | Type | What it tests | Metric |
|---|---|---|---|---|
| `badnets` | BadNets | removal | baseline: one client, data poison, static patch | asr |
| `dba_partial` | DBA (4 colluders, unlearn 1) | removal | a backdoor split across clients, unlearning only one | asr |
| `dba_detected` | DBA (4 colluders, unlearn all) | removal | the same backdoor with all 4 attackers unlearned | asr |
| `neurotoxin` | Neurotoxin | removal | a backdoor built to survive later training | asr |
| `edgecase` | Edge-Case (CIFAR tail samples) | removal | a backdoor on a rare class subpopulation held out of honest clients | edgecase_asr |
| `badfu` | BadFU | exploitation | a hidden backdoor that the unlearning request turns on | asr (resurgence-graded) |
| `fedmua` | FedMUA | exploitation | an unlearning request that makes the model misclassify a chosen class | misclassification |

`dba_partial` unlearns only one of the four attacking clients; `dba_detected` unlearns all four. Comparing the two shows whether unlearning one client can clear a backdoor spread across several. It cannot: the other three contributions stay in the retain set.

`edgecase` backdoors the rarest slice of a class, the tail, meaning the source-class images that a clean reference model (`reference_model.pt`, trained once on CIFAR) is least confident about. Those are the genuinely atypical members, the ones honest training cannot easily pull back, which is what lets the backdoor persist. The same reference model picks the test-set tail, so the trained and scored tails are the same concept. Those exact images are held out of every honest client and given only to the attacker, relabeled to the target. If honest clients kept them with their true labels, normal training would overwrite the backdoor every round and it would never persist. Holding them out matches the original Attack of the Tails, where the edge data is out of distribution and honest clients never see it, adapted here to stay inside CIFAR-10. The control run omits the tail too, so it stands for a world where the attacker never existed.

## 6. Metrics and outputs

Each run writes `metrics_<row>.json`. For every scorer it records:

- `pre_unlearn_<m>` and `post_unlearn_<m>`: before and after unlearning
- `rfs_<m>`: the retrain-from-scratch control run
- `post_resurge_<m>`: BadFU only. ASR after a short benign fine-tune that checks for a hidden backdoor.
- `efficiency_*`: cost numbers, measured by the suite itself (not a scorer, since cost is about the process). These are `unlearn_wall_s`, `unlearn_peak_mem_mb`, `rfs_wall_s`, `rfs_peak_mem_mb`, `rfs_comm_rounds`, `storage_overhead_bytes`, and the derived `wall_speedup`. The optional `unlearn_comm_rounds`, `unlearn_steps`, `unlearn_samples`, and `comm_round_fraction` appear only when an unlearner fills `context.cost`.

When `seeds` has more than one value, the whole pipeline reruns once per seed. Each metric key then holds the mean across seeds, and `<metric>_std` and `<metric>_seeds` carry the spread and the raw per-seed values. With one seed the file is unchanged.

`benchmark.py` combines the per-row files into `benchmark_metrics.json`: one `gap_to_rfs` number per attack. It does not average across different metrics.

## 7. Pipeline flow

```
main.py  --mode {test|single|attack|benchmark|aggregate}
  run_single_attack(name)
    attack_config(name)              #roster row to ExperimentConfig
    run_experiment(cfg):
      for seed in cfg.seeds:           #rerun the whole pipeline per seed
        1. load CIFAR-10, build threat model, scorers, Benchmarker
        2. federated_train (attacked)  OR  load cached weights  ->  global model
             (FUDGEStrategy forces the malicious clients every round)
        3. RFS control run: federated_train with the attack off          [timed]
        4. build forget and retain loaders (set by the threat model)
        5. unlearner.unlearn(...)                                        [timed]
        6. Benchmarker report (pre/post/rfs) + optional resurgence probe + efficiency
      7. aggregate seeds (mean, std) -> write metrics_<row>.json
  benchmark.py: combine metrics_*.json  ->  benchmark_metrics.json
```

## 8. Key design decisions

- Backdoor removal means client-level unlearning: forget the malicious client's contribution. This is horizontal FU.
- RFS is the control run. It is a phase in the runner (retrain with the attack off), not a `BaseUnlearner`, because it retrains instead of cleaning an existing model.
- `history_cache` (per-round client updates) is off by default and turned on with `cache_history`, only for FedEraser-style unlearners that read it. PGA does not, so it stays empty and storage overhead is zero. Caching it every round otherwise grows to tens of GB and can OOM the workers.
- FedAvg is fixed on purpose, so the backdoor gets in and the unlearner has something to remove. Swapping in robust aggregation would test prevention instead, and mix the two stages.
- Multiple seeds give error bars. `seeds` reruns the whole pipeline and reports mean and std. A single number is noisy: the per-round attack score swings run to run. The 50-client partition stays fixed at seed 42; only the training randomness (model init, client sampling, batch order) varies, so the spread reflects run-to-run noise on a fixed benchmark, not a different data split each time.
- An attack can hold a subpopulation out of the honest clients. A threat model reports `holdout_indices()`, and those samples are dropped from every honest partition, the retain set, and the control run, so they belong only to the attacker. Edge-Case uses this for durability: if the honest clients kept the tail images with true labels, they would scrub the backdoor each round. The default is empty, so other attacks are unaffected.

## 9. Source layout

```
src/
  main.py             entrypoint, run modes
  config.py           ExperimentConfig (all settings) + fixed constants
  registry.py         decorators + lazy import of built-ins
  runner.py           run_experiment: train, RFS, unlearn, audit, report
  benchmark.py        roster + per-attack combine
  training.py         federated_train (shared by attacked run and RFS)
  client.py           FUDGEClient (local SGD, poison injection)
  strategies/         FUDGEStrategy (FedAvg + forced malicious clients + history cache)
  models/             build_model (fixed ResNet-18)
  datasets/           Dirichlet partitioner + in-memory backdoor dataset
  threat_models/      BaseThreatModel + 6 attacks
  unlearning/         BaseUnlearner + pga + rfs control
  audit/              Benchmarker + scorers
```
