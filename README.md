# FUDGE-Suite

**By: [Elliot Hong](https://www.linkedin.com/in/david-elliot-hong)**

A modular, standardized benchmark for **federated unlearning (FU)** algorithms — measuring whether an unlearner is accurate, private, efficient, and **secure** (both able to *remove* implanted backdoors and able to *resist* having the deletion request weaponized against it).

Emerging work like [BadFU](https://arxiv.org/pdf/2508.15541) shows unlearning requests can be exploited to backdoor a model, while the field still [lacks a standardized evaluation framework](https://dl.acm.org/doi/epdf/10.1145/3679014) for horizontal FU. FUDGE-Suite fills that gap.

> **New here?** Read [ARCHITECTURE.md](ARCHITECTURE.md) for the mental model — fixed vs pluggable axes, the threat roster, metrics, and pipeline flow.

---

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
| `single` | One run from default `ExperimentConfig` |
| `attack --attack <name>` | One roster row end-to-end → `metrics_<name>.json` |
| `benchmark` | Full roster, serial → `benchmark_metrics.json` |
| `aggregate` | Fold existing `metrics_*.json` into the summary profile |

On a SLURM cluster, `submit_benchmark.sh` runs the roster as an array (`--array=0-6`); use `sbatch --array=0` for a single row.

**Typical workflow:**
```bash
uv run python -m src.main --mode test                     # smoke test (2 rounds)
uv run python -m src.main --mode attack --attack badnets  # one row -> metrics_badnets.json
sbatch submit_benchmark.sh                                # full roster on SLURM (array 0-6)
uv run python -m src.main --mode aggregate                # fold metrics_*.json -> benchmark_metrics.json
```

## Roster

`badnets`, `dba_partial`, `dba_detected`, `neurotoxin`, `edgecase` (eradication) · `badfu`, `fedmua` (exploitation). See [ARCHITECTURE.md §5](ARCHITECTURE.md).

## Configurable options

All knobs live in `ExperimentConfig` (`src/config.py`); roster rows override per-attack fields in `src/benchmark.py`.

**Pluggable axes**
| Option | Default | |
|---|---|---|
| `unlearner` | `pga` | FU algorithm under test |
| `unlearner_args` | `{epochs, lr, projection_radius, retain_enabled}` | algorithm hyperparams |
| `threat_model` / `threat_model_args` | per row | attack + its params |
| `scorers` | `[accuracy, asr]` | `asr` / `edgecase_asr` / `misclassification` |

**Attack strength**
| Option | Default | |
|---|---|---|
| `target_label` | 0 | backdoor target class |
| `poison_ratio` | 0.5 | fraction of malicious client data poisoned |
| `amplification_factor` | 4.0 | malicious update scaling (model-replacement) |
| `patch_size` | 3 | trigger patch size |

**Unlearn scope & probes**
| Option | Default | |
|---|---|---|
| `malicious_client_id` | `"0"` | attacker client |
| `unlearn_client_id` / `unlearn_client_ids` | `"0"` / `None` | scope (list for multi-client / DBA-detected) |
| `resurgence_probe` / `_steps` / `_lr` | `False` / 100 / 0.01 | latent-backdoor probe (BadFU) |

**Training substrate** (tuned within the locked 50-round budget)
| Option | Default | |
|---|---|---|
| `local_epochs` | 8 | client local epochs/round |
| `client_lr` / `lr_cosine` | 0.03 / `True` | peak LR + cosine decay to 0 |
| `seed` | 0 | reproducibility |

**Run control**
| Option | Default | |
|---|---|---|
| `run_rfs_baseline` | `True` | compute the retrain-from-scratch control |
| `use_cached_weights` / `weights_cache_path` | `False` / — | skip training, load cached global weights |
| `output_path` | `run_metrics.json` | per-run metrics file |

## Outputs

`metrics_<row>.json` reports `pre_unlearn_*`, `post_unlearn_*`, `rfs_*`, optional `post_resurge_*` (BadFU), and `efficiency_*` (wall/mem/storage + RFS-relative speedup). `benchmark_metrics.json` is the per-attack `gap_to_rfs` profile.
