import argparse
from dataclasses import replace

from src.config import ExperimentConfig, test_config
from src.runner import run_experiment
from src.benchmark import run_benchmark, run_single_attack, aggregate_from_files


#pick the unlearner, federaser needs its own args and the per-round cache
def _base_config(unlearner):
    if unlearner is None or unlearner == "pga":
        return None
    overrides = {"unlearner": unlearner}
    if unlearner == "federaser":
        overrides["unlearner_args"] = {"calib_steps": 10, "calib_lr": 0.01, "calib_interval": 1}
        overrides["cache_history"] = True
    return replace(ExperimentConfig(), **overrides)


#test run, single attack, full roster, one array attack, or fold metrics
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        choices=["test", "single", "benchmark", "attack", "aggregate"],
                        default="single")
    parser.add_argument("--attack", default=None, help="threat model name for mode attack")
    parser.add_argument("--unlearner", default=None, help="unlearner override, e.g. federaser")
    args = parser.parse_args()

    if args.mode == "test":
        run_experiment(test_config())
    elif args.mode == "benchmark":
        run_benchmark()
    elif args.mode == "attack":
        if args.attack is None:
            parser.error("--mode attack requires --attack <name>")
        run_single_attack(args.attack, _base_config(args.unlearner))
    elif args.mode == "aggregate":
        aggregate_from_files()
    else:
        run_experiment(ExperimentConfig())


if __name__ == "__main__":
    main()
