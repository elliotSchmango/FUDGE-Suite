import json
from dataclasses import replace

from src.config import ExperimentConfig
from src.runner import run_experiment


#each attack carries its own args and scorers
def default_roster():
    return [
        {"threat_model": "badnets", "threat_model_args": {},
         "scorers": ["accuracy", "asr"]},
        {"threat_model": "dba", "threat_model_args": {"num_saboteurs": 4},
         "scorers": ["accuracy", "asr"]},
        {"threat_model": "neurotoxin", "threat_model_args": {"mask_ratio": 0.1},
         "scorers": ["accuracy", "asr"]},
        {"threat_model": "edgecase", "threat_model_args": {"source_class": 1, "tail_fraction": 0.1},
         "scorers": ["accuracy", "edgecase_asr"]},
        {"threat_model": "badfu", "threat_model_args": {"camou_ratio": 0.2},
         "scorers": ["accuracy", "asr"]},
        {"threat_model": "fedmua", "threat_model_args": {"victim_class": 0, "num_requests": 20},
         "scorers": ["accuracy", "misclassification"]},
    ]


#distinction between asr vs misclassification threat types
def _attack_metric(report):
    for m in ("asr", "misclassification"):
        if f"post_unlearn_{m}" in report:
            return m
    return None


#gap-to-rfs
    rows = {}
    for attack, report in results.items():
        m = _attack_metric(report)
        post = report.get(f"post_unlearn_{m}")
        rfs = report.get(f"rfs_{m}")
        gap = (post - rfs) if (post is not None and rfs is not None) else None
        rows[attack] = {
            "metric": m,
            "pre": report.get(f"pre_unlearn_{m}"),
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


#run the full roster for one unlearner
def run_benchmark(base_config=None, roster=None, output_path="benchmark_metrics.json"):
    base_config = base_config or ExperimentConfig()
    roster = roster or default_roster()

    results = {}
    for spec in roster:
        attack = spec["threat_model"]
        cfg = replace(base_config, output_path=f"metrics_{attack}.json", **spec)
        print(f"\n===== benchmark attack: {attack} (unlearner={cfg.unlearner}) =====")
        results[attack] = run_experiment(cfg)

    aggregate = _aggregate(results)
    with open(output_path, "w") as f:
        json.dump(aggregate, f, indent=4)
    print(f"\nbenchmark summary saved to {output_path}")
    return aggregate
