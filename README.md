# FUDGE-Suite

**By: [Elliot Hong](https://www.linkedin.com/in/david-elliot-hong)**

A modular, standardized benchmark for federated unlearning (FU) algorithms. It measures whether an unlearner is accurate, private, efficient, and secure (able to remove planted backdoors, and able to resist attacks that abuse the deletion request).

Recent works like [BadFU](https://arxiv.org/pdf/2508.15541) shows that unlearning requests can be used to backdoor a model, while the field still [lacks a standardized evaluation framework](https://dl.acm.org/doi/epdf/10.1145/3679014) for horizontal FU. FUDGE-Suite fills that gap. For how the suite works inside, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Install

```bash
uv sync
```

First run downloads CIFAR-10 to `data/` and generates the seeded 50-client partition (`src/datasets/partitions.json`).

## Run

```bash
uv run python -m src.main --mode <mode> [--attack <name>]
```

| Mode | What it does |
|---|---|
| `test` | Fast end-to-end smoke run (2 rounds, BadNets) |
| `single` | One run from the default `ExperimentConfig` |
| `attack --attack <name>` | One roster row end to end, writes `metrics_<name>.json` |
| `benchmark` | Full roster in one process, writes `benchmark_metrics.json` |
| `aggregate` | Combine existing `metrics_*.json` into the summary |

On a SLURM cluster, `submit_benchmark.sh` runs the roster as an array (`--array=0-6`); use `sbatch --array=0` for a single row.

**Typical workflow:**
```bash
uv run python -m src.main --mode test                     #smoke test (2 rounds)
uv run python -m src.main --mode attack --attack badnets  #one row, writes metrics_badnets.json
sbatch submit_benchmark.sh                                #full roster on SLURM (array 0-6)
uv run python -m src.main --mode aggregate                #combine metrics_*.json into benchmark_metrics.json
```

## Roster

`badnets`, `dba_partial`, `dba_detected`, `neurotoxin` (removal), plus `badfu`, `fedmua` (exploitation). See [ARCHITECTURE.md](ARCHITECTURE.md).

## Configurable options

All settings live in `ExperimentConfig` (`src/config.py`); roster rows override per-attack fields in `src/benchmark.py`.

**Pluggable parts**
| Option | Default | |
|---|---|---|
| `unlearner` | `pga` | FU algorithm under test |
| `unlearner_args` | `{epochs, lr, projection_radius, retain_enabled}` | algorithm settings |
| `threat_model` / `threat_model_args` | per row | attack and its settings |
| `scorers` | `[accuracy, asr]` | `asr` or `misclassification` |

**Attack strength**
| Option | Default | |
|---|---|---|
| `target_label` | 0 | backdoor target class |
| `poison_ratio` | 0.5 | fraction of the malicious client's data that is poisoned |
| `amplification_factor` | 4.0 | malicious update scaling |
| `patch_size` | 3 | trigger patch size |

**Unlearn scope and probes**
| Option | Default | |
|---|---|---|
| `malicious_client_id` | `"0"` | attacker client |
| `unlearn_client_id` / `unlearn_client_ids` | `"0"` / `None` | scope (list for multi-client, e.g. dba_detected) |
| `resurgence_probe` / `_steps` / `_lr` | `False` / 100 / 0.01 | hidden-backdoor probe (BadFU) |

**Training settings** (tuned within the fixed 50-round budget)
| Option | Default | |
|---|---|---|
| `local_epochs` | 8 | client local epochs per round |
| `client_lr` / `lr_cosine` | 0.03 / `True` | peak LR and cosine decay to 0 |
| `seeds` | `[0]` | seed list. More than one reruns the pipeline and reports mean and std |

**Run control**
| Option | Default | |
|---|---|---|
| `run_rfs_baseline` | `True` | run the retrain-from-scratch control |
| `cache_history` | `False` | store per-round updates, only for FedEraser-style unlearners |
| `use_cached_weights` / `weights_cache_path` | `False` / (none) | skip training, load cached global weights |
| `output_path` | `run_metrics.json` | per-run metrics file |

## Outputs

`metrics_<row>.json` reports `pre_unlearn_*`, `post_unlearn_*`, `rfs_*`, optional `post_resurge_*` (BadFU), and `efficiency_*` (time, memory, storage, and speed relative to RFS). `benchmark_metrics.json` is the `gap_to_rfs` number for each attack.

With more than one seed, each metric key holds the mean across seeds, and two extra keys appear per metric: `<metric>_std` (spread) and `<metric>_seeds` (the per-seed values). With one seed the file is unchanged.
