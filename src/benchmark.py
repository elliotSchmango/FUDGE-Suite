import os
import json
from dataclasses import replace

from src.config import ExperimentConfig
from src.runner import run_experiment


#roster rows
def default_roster():
    return [
        {"name": "badnets", "threat_model": "badnets", "threat_model_args": {},
         "scorers": ["accuracy", "asr"]},
        {"name": "dba_partial", "threat_model": "dba", "threat_model_args": {"num_saboteurs": 4},
         "scorers": ["accuracy", "asr"]},
        {"name": "dba_detected", "threat_model": "dba", "threat_model_args": {"num_saboteurs": 4},
         "scorers": ["accuracy", "asr"], "unlearn_client_ids": ["0", "1", "2", "3"]},
        {"name": "neurotoxin", "threat_model": "neurotoxin", "threat_model_args": {"mask_ratio": 0.1},
         "scorers": ["accuracy", "asr"]},
        {"name": "badfu", "threat_model": "badfu", "threat_model_args": {"camou_ratio": 0.2},
         "scorers": ["accuracy", "asr"], "resurgence_probe": True},
        {"name": "fedmua", "threat_model": "fedmua", "threat_model_args": {"victim_class": 0, "num_requests": 20},
         "scorers": ["accuracy", "misclassification"]},
    ]


#asr vs misclassification metric
def _attack_metric(report):
    for m in ("asr", "misclassification"):
        if f"post_unlearn_{m}" in report:
            return m
    return None


#gap-to-rfs for each attack
def _aggregate(results):
    rows = {}
    for attack, report in results.items():
        m = _attack_metric(report)
        #resurgence is graded post when present
        resurge = report.get(f"post_resurge_{m}")
        post_unlearn = report.get(f"post_unlearn_{m}")
        post = resurge if resurge is not None else post_unlearn
        rfs = report.get(f"rfs_{m}")
        gap = (post - rfs) if (post is not None and rfs is not None) else None
        rows[attack] = {
            "metric": m,
            "pre": report.get(f"pre_unlearn_{m}"),
            "post_unlearn": post_unlearn,
            "post_resurge": resurge,
            "post": post,
            "rfs": rfs,
            "gap_to_rfs": gap,
            "post_accuracy": report.get("post_unlearn_accuracy"),
        }

    gaps = [r["gap_to_rfs"] for r in rows.values() if r["gap_to_rfs"] is not None]
    accs = [r["post_accuracy"] for r in rows.values() if r["post_accuracy"] is not None]
    summary = {
        "mean_gap_to_rfs": sum(gaps) / len(gaps) if gaps else None,
        "mean_post_accuracy": sum(accs) / len(accs) if accs else None,
        "num_attacks": len(rows),
    }
    return {"per_attack": rows, "summary": summary}


#per-row config with own output and cache
def attack_config(attack_name, base_config=None, roster=None):
    base_config = base_config or ExperimentConfig()
    roster = roster or default_roster()
    spec = next((s for s in roster if s["name"] == attack_name), None)
    if spec is None:
        raise ValueError(f"unknown attack {attack_name}")
    #remaining keys map to config fields
    overrides = {k: v for k, v in spec.items() if k != "name"}
    #suffix non-pga unlearners so they get their own files, never clobber pga results
    suffix = "" if base_config.unlearner == "pga" else f"_{base_config.unlearner}"
    return replace(
        base_config,
        output_path=f"results/metrics_{attack_name}{suffix}.json",
        weights_cache_path=f"cache_{attack_name}{suffix}.npz",
        **overrides,
    )


#run one attack end to end
def run_single_attack(attack_name, base_config=None):
    cfg = attack_config(attack_name, base_config)
    print(f"\n===== attack {attack_name} (unlearner={cfg.unlearner}) =====")
    return run_experiment(cfg)


#fold per-attack metrics into summary
def aggregate_from_files(roster=None, output_path="benchmark_metrics.json"):
    roster = roster or default_roster()
    results = {}
    for spec in roster:
        attack = spec["name"]
        path = f"results/metrics_{attack}.json"
        if os.path.exists(path):
            with open(path) as f:
                results[attack] = json.load(f)
        else:
            print(f"warning, missing {path}")

    aggregate = _aggregate(results)
    with open(output_path, "w") as f:
        json.dump(aggregate, f, indent=4)
    print(f"\nbenchmark summary saved to {output_path}")
    return aggregate


#serial roster run
def run_benchmark(base_config=None, roster=None, output_path="benchmark_metrics.json"):
    base_config = base_config or ExperimentConfig()
    roster = roster or default_roster()

    results = {}
    for spec in roster:
        results[spec["name"]] = run_single_attack(spec["name"], base_config)

    aggregate = _aggregate(results)
    with open(output_path, "w") as f:
        json.dump(aggregate, f, indent=4)
    print(f"\nbenchmark summary saved to {output_path}")
    return aggregate
