import argparse

from src.config import ExperimentConfig, test_config
from src.runner import run_experiment
from src.benchmark import run_benchmark, run_single_attack, aggregate_from_files


#test run, single attack, full roster, one array attack, or fold metrics
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        choices=["test", "single", "benchmark", "attack", "aggregate"],
                        default="single")
    parser.add_argument("--attack", default=None, help="threat model name for mode attack")
    args = parser.parse_args()

    if args.mode == "test":
        run_experiment(test_config())
    elif args.mode == "benchmark":
        run_benchmark()
    elif args.mode == "attack":
        if args.attack is None:
            parser.error("--mode attack requires --attack <name>")
        run_single_attack(args.attack)
    elif args.mode == "aggregate":
        aggregate_from_files()
    else:
        run_experiment(ExperimentConfig())


if __name__ == "__main__":
    main()
